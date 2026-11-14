"""Kinematics and IK on Pinocchio (C++) for any fixed-base serial arm described by a URDF. The solver is the
data_viewer console's own: the piper-IK Pinocchio backend, copied unchanged (piper-IK 0.2.0, MIT, Copyright (c) 2026
kelo020304; see LICENSE). The bundled model is model/piper_umi.urdf (Piper + UMI gripper).

  - model: the URDF reduced to the arm joints. By default these are the movable joints on the chain from the root to
    the TCP frame; every other joint (fingers) is locked at 0. IK frame = the child link of the last arm joint (Piper:
    link6, the flange). Revolute and prismatic joints with limits; any number of them.
  - IK: bounded damped least squares, weights (1,1,1,.3,.3,.3), bounded least squares (scipy lsq_linear 'bvls',
    bit-identical copy), <= 0.2 rad per step, backtracking line search; the target is first multiplied by inv(TCP),
    as piper-IK does
Pinocchio Data is mutable: one KinematicModel per thread (ArmIK guards its own with a lock).
"""
import ctypes
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def _import_pinocchio():
    """Import Pinocchio (pip package `pin`). The extension needs `libeigenpy.so`; when another eigenpy comes first
    on LD_LIBRARY_PATH (ROS noetic's, built against Python 3.8's boost) the loader would take that one and the
    import fails with "undefined symbol: _Py_tracemalloc_config". Loading the wheel's own libeigenpy.so by
    absolute path first makes the loader reuse it; nothing else changes."""
    spec = importlib.util.find_spec('pinocchio')
    if spec is None or spec.origin is None:
        raise ImportError('piper_ik needs Pinocchio: pip install pin==3.7.0')
    lib = Path(spec.origin).resolve().parents[3] / 'libeigenpy.so'          # .../cmeel.prefix/lib/libeigenpy.so (pip wheel layout)
    if lib.is_file():
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    import pinocchio
    return pinocchio


pin = _import_pinocchio()
ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 7))        # the Piper's arm joints (what the bundled model resolves to)
# The TCP is the console's own 4x4 matrix (model/tcp.json), used as is: the URDF can only write it as xyz / rpy, which
# rebuilds it to 8e-16, and only the exact matrix makes the IK bit-identical to the console's.
CONSOLE_TCP = np.array(json.loads((Path(__file__).resolve().parent / 'model' / 'tcp.json').read_text())['matrix'], float)


def _frame_id(model, name, what):
    if not model.existFrame(name):
        raise ValueError(f'{what} {name!r} is not in the URDF; frames: {[f.name for f in model.frames[1:]]}')
    return model.getFrameId(name)


class KinematicModel:
    """A URDF as a Pinocchio model of its arm joints. Same attributes as piper-IK's KinematicModel: model, data,
    frame (the IK frame), lower / upper (the solver reads them), fk(q) -> SE3; plus tcp (IK frame -> TCP, 4x4).

    tcp_frame: the frame whose pose the caller asks for (fingertip TCP), fixed on the last arm link.
    frame: the IK frame; default: the child link of the joint that carries tcp_frame (Piper: link6).
    joints: names of the arm joints; default: the movable joints on the chain from the root to tcp_frame. Every other
        joint is locked at 0.
    tcp: IK frame -> TCP (4x4) in place of the one the URDF gives."""

    def __init__(self, urdf, frame=None, tcp_frame='tool0', joints=None, tcp=None):
        full = pin.buildModelFromUrdf(str(urdf))
        if joints is None:                                                       # the chain root -> TCP frame
            chain, j = [], full.frames[_frame_id(full, tcp_frame, 'tcp_frame')].parentJoint
            while j > 0:
                chain.append(j); j = full.parents[j]
            keep = set(chain)
        else:
            missing = [n for n in joints if not full.existJointName(n)]
            if missing:
                raise ValueError(f'joints {missing} are not in the URDF; joints: {list(full.names[1:])}')
            keep = {full.getJointId(n) for n in joints}
        lock = [i for i in range(1, full.njoints) if i not in keep]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full)) if lock else full
        bad = [self.model.names[j] for j in range(1, self.model.njoints) if self.model.joints[j].nq != 1 or self.model.joints[j].nv != 1]
        if self.model.nq < 1 or bad:
            raise ValueError(f'arm joints must be revolute or prismatic (continuous, spherical and floating joints are not supported): {bad or "none found"}')
        self.data = self.model.createData()
        t = _frame_id(self.model, tcp_frame, 'tcp_frame') if tcp is None or frame is None else None
        if frame is None:                                                        # the link the last arm joint moves
            jt = self.model.frames[t].parentJoint; jf = self.model.getFrameId(self.model.names[jt], pin.FrameType.JOINT)
            frame = next(f.name for f in self.model.frames if f.type == pin.FrameType.BODY and f.parentFrame == jf)
        self.frame = _frame_id(self.model, frame, 'IK frame')
        self.lower = self.model.lowerPositionLimit.copy(); self.upper = self.model.upperPositionLimit.copy()
        if not (np.isfinite(self.lower).all() and np.isfinite(self.upper).all() and np.all(self.lower < self.upper)):
            raise ValueError(f'every arm joint needs finite limits with lower < upper: {self.lower} .. {self.upper}')
        f = self.model.frames[self.frame]
        if tcp is not None:
            tcp = np.array(tcp, float)
            if tcp.shape != (4, 4):
                raise ValueError('tcp must be a 4x4 matrix (IK frame -> TCP)')
            self.tcp = tcp
            return
        t = self.model.frames[t]
        if f.parentJoint != t.parentJoint:
            raise ValueError(f'{tcp_frame} must be rigidly attached to {frame}')
        tcp = (f.placement.inverse() * t.placement).homogeneous.copy()               # IK frame -> TCP
        self.tcp = CONSOLE_TCP.copy() if np.abs(tcp - CONSOLE_TCP).max() < 1e-12 else tcp    # bundled model: the console's exact matrix

    @property
    def joint_names(self):
        return list(self.model.names[1:])

    def fk(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frame].copy()


@dataclass(frozen=True)
class SolverConfig:
    """piper-IK PinocchioConfig defaults, unchanged."""
    weights: tuple = (1., 1., 1., .3, .3, .3)
    damping: float = 1e-7
    singularity_threshold: float = .01
    seed_weight: float = 1e-8
    max_joint_step: float = .2
    max_iterations: int = 40
    position_tolerance: float = 1e-4        # m, at the TCP
    orientation_tolerance: float = 1e-3     # rad, at the TCP


def _kkt_optimality(g, on_bound):
    g_kkt = g * on_bound
    free_set = on_bound == 0
    g_kkt[free_set] = np.abs(g[free_set])
    return np.max(g_kkt)


def bvls(A, b, lb, ub, tol=1e-10):
    """scipy.optimize.lsq_linear(A, b, (lb, ub), method='bvls', tol=tol).x for a dense A with lb < ub: the same NumPy
    calls in the same order (scipy/optimize/_lsq/lsq_linear.py and bvls.py, BSD-3-Clause), without the input checks,
    printing and result bookkeeping, so the solution is bit-identical and about twice as fast to get."""
    x_lsq = np.linalg.lstsq(A, b, rcond=-1)[0]
    if np.all((x_lsq >= lb) & (x_lsq <= ub)):
        return x_lsq
    m, n = A.shape
    x = x_lsq.copy()
    on_bound = np.zeros(n)
    mask = x <= lb
    x[mask] = lb[mask]
    on_bound[mask] = -1
    mask = x >= ub
    x[mask] = ub[mask]
    on_bound[mask] = 1
    free_set = on_bound == 0
    active_set = ~free_set
    free_set, = np.nonzero(free_set)
    r = A.dot(x) - b
    cost = 0.5 * np.dot(r, r)
    g = A.T.dot(r)
    iteration = 0
    while free_set.size > 0:                                     # initialization: least squares on the free set
        iteration += 1
        A_free = A[:, free_set]
        b_free = b - A.dot(x * active_set)
        z = np.linalg.lstsq(A_free, b_free, rcond=None)[0]
        lbv = z < lb[free_set]
        ubv = z > ub[free_set]
        v = lbv | ubv
        if np.any(lbv):
            ind = free_set[lbv]
            x[ind] = lb[ind]
            active_set[ind] = True
            on_bound[ind] = -1
        if np.any(ubv):
            ind = free_set[ubv]
            x[ind] = ub[ind]
            active_set[ind] = True
            on_bound[ind] = 1
        ind = free_set[~v]
        x[ind] = z[~v]
        r = A.dot(x) - b
        cost = 0.5 * np.dot(r, r)
        g = A.T.dot(r)
        if np.any(v):
            free_set = free_set[~v]
        else:
            break
    max_iter = n + iteration
    termination_status = None
    optimality = _kkt_optimality(g, on_bound)
    for iteration in range(iteration, max_iter):                 # main loop
        if optimality < tol:
            termination_status = 1
        if termination_status is not None:
            break
        move_to_free = np.argmax(g * on_bound)
        on_bound[move_to_free] = 0
        while True:
            free_set = on_bound == 0
            active_set = ~free_set
            free_set, = np.nonzero(free_set)
            x_free = x[free_set]
            lb_free = lb[free_set]
            ub_free = ub[free_set]
            A_free = A[:, free_set]
            b_free = b - A.dot(x * active_set)
            z = np.linalg.lstsq(A_free, b_free, rcond=None)[0]
            lbv, = np.nonzero(z < lb_free)
            ubv, = np.nonzero(z > ub_free)
            v = np.hstack((lbv, ubv))
            if v.size > 0:
                alphas = np.hstack((lb_free[lbv] - x_free[lbv], ub_free[ubv] - x_free[ubv])) / (z[v] - x_free[v])
                i = np.argmin(alphas)
                i_free = v[i]
                alpha = alphas[i]
                x_free *= 1 - alpha
                x_free += alpha * z
                x[free_set] = x_free
                if i < lbv.size:
                    on_bound[free_set[i_free]] = -1
                else:
                    on_bound[free_set[i_free]] = 1
            else:
                x_free = z
                x[free_set] = x_free
                break
        r = A.dot(x) - b
        cost_new = 0.5 * np.dot(r, r)
        cost_change = cost - cost_new
        if cost_change < tol * cost:
            termination_status = 2
        cost = cost_new
        g = A.T.dot(r)
        optimality = _kkt_optimality(g, on_bound)
    return x


def solve_flange(k, target, seed, c=SolverConfig()):
    """piper-IK PinocchioBackend.solve, unchanged except that the regularization block is n x n for n arm joints:
    IK-frame target (4x4) from `seed`; never leaves [k.lower, k.upper].

    Each iteration linearizes at q (frame Jacobian J, 6-d pose error e = log6) and solves one bounded linear least
    squares problem for the step dq:
        min || [W J; r I] dq - [-W e; -s (q - q_seed) / r] ||^2,   max(lower - q, -0.2) <= dq <= min(upper - q, 0.2)
    W = weights, r^2 = damping (raised near singularities) + s, s = seed_weight. Then a backtracking line search."""
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
        A = np.vstack([J, reg * np.eye(len(q))])                # Piper: eye(6), as piper-IK writes it
        rhs = np.r_[-e, -c.seed_weight * (q - initial) / reg]
        lower = np.maximum(k.lower - q, -c.max_joint_step)
        upper = np.minimum(k.upper - q, c.max_joint_step)
        free = upper - lower > 1e-12
        dq = lower.copy()
        if np.any(free):
            dq[free] = bvls(A[:, free], rhs - A[:, ~free] @ dq[~free], lower[free], upper[free], tol=1e-10)
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
