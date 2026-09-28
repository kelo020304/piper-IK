"""Bounded damped least-squares backend and Piper kinematic model."""
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.optimize import lsq_linear

ASSETS = Path(__file__).resolve().parent / 'assets'


class KinematicModel:
    def __init__(self, urdf=None, frame='tool0'):
        self.model = pin.buildModelFromUrdf(str(urdf or ASSETS / 'piper.urdf'))
        if self.model.nq != 6 or self.model.nv != 6:
            raise ValueError('PiperIK requires six scalar joints')
        if not self.model.existFrame(frame):
            raise ValueError(f'Unknown end-effector frame: {frame}')
        self.data = self.model.createData()
        self.frame = self.model.getFrameId(frame)
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()

    def fk(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frame].copy()


class PinocchioBackend:
    def __init__(self, model, config):
        self.kinematics, self.config = model, config

    def solve(self, target, seed):
        k, c = self.kinematics, self.config
        target = pin.SE3(target)
        q, initial = seed.copy(), seed.copy()
        weight = np.asarray(c.weights)
        for _ in range(c.max_iterations):
            pose = k.fk(q)
            delta = pose.inverse() * target
            pe = np.linalg.norm(pose.translation - target.translation)
            oe = np.linalg.norm(pin.log3(pose.rotation.T @ target.rotation))
            if pe < c.position_tolerance and oe < c.orientation_tolerance:
                break
            error = pin.log6(delta).vector
            jac = pin.computeFrameJacobian(k.model, k.data, q, k.frame, pin.ReferenceFrame.LOCAL)
            J = weight[:, None] * (-pin.Jlog6(delta.inverse()) @ jac)
            e = weight * error
            sigma = np.linalg.svd(J, compute_uv=False)[-1]
            damping = c.damping + max(0., c.singularity_threshold - sigma)**2
            reg = np.sqrt(damping + c.seed_weight)
            A = np.vstack([J, reg * np.eye(6)])
            rhs = np.r_[-e, -c.seed_weight * (q - initial) / reg]
            lower = np.maximum(k.lower - q, -c.max_joint_step)
            upper = np.minimum(k.upper - q, c.max_joint_step)
            free = upper - lower > 1e-12
            dq = lower.copy()
            if np.any(free):
                dq[free] = lsq_linear(
                    A[:, free], rhs - A[:, ~free] @ dq[~free],
                    bounds=(lower[free], upper[free]), method='bvls', tol=1e-10,
                ).x
            accepted = False
            for alpha in (1., .5, .25, .125, .0625):
                trial = np.clip(q + alpha * dq, k.lower, k.upper)
                er = weight * pin.log6(k.fk(trial).inverse() * target).vector
                if er @ er < e @ e - 1e-16:
                    q, accepted = trial, True
                    break
            if not accepted:
                break
        return q
