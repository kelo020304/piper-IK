"""ROS-free SE(3) IK; warm start and hard trajectory increment bounds.

Uses Pinocchio log6/frame Jacobians, as in AgileX QuestArmTeleop's IK core.
Unlike the ROS node, optimization here is bounded damped least squares and
trajectory bounds are enforced explicitly, not only by a smoothness cost.
"""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pinocchio as pin
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]/'assets'


@dataclass
class IKResult:
    q: np.ndarray
    position_error: float
    orientation_error: float
    converged: bool
    iterations: int
    seconds: float


class PinIK:
    def __init__(self, urdf=ROOT/'models/piper.urdf', frame='tool0'):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.frame = self.model.getFrameId(frame)
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        self.weight = np.array([1., 1., 1., .3, .3, .3])
        self.q_previous = None
        self.velocity = np.zeros(self.model.nv)

    def fk(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frame].copy()

    def solve(self, target, seed, bounds=None, max_iter=100, position_tol=1e-4, orientation_tol=1e-3):
        start = time.perf_counter()
        target = target if isinstance(target, pin.SE3) else pin.SE3(target)
        lo, hi = (self.lower, self.upper) if bounds is None else bounds
        q = np.clip(np.asarray(seed, dtype=float), lo, hi)
        initial = q.copy()
        for it in range(max_iter):
            pose = self.fk(q)
            delta = pose.inverse() * target
            error = pin.log6(delta).vector
            pe = np.linalg.norm(pose.translation-target.translation)
            oe = np.linalg.norm(pin.log3(pose.rotation.T@target.rotation))
            if pe < position_tol and oe < orientation_tol: break
            jac = pin.computeFrameJacobian(self.model,self.data,q,self.frame,pin.ReferenceFrame.LOCAL)
            jac = -pin.Jlog6(delta.inverse()) @ jac
            J = self.weight[:,None]*jac; e=self.weight*error
            # Damping regularizes singular configurations; tiny seed regularization
            # breaks equivalent solutions in favor of the previous trajectory state.
            singular = np.linalg.svd(J,compute_uv=False)[-1]
            damping = 1e-7 + max(0., .01-singular)**2
            # Solve inside the joint increment box. Clipping an unconstrained
            # Newton step can stall on tiny pure-rotation keyboard requests.
            A=np.vstack([J,np.sqrt(damping+1e-8)*np.eye(self.model.nv)])
            rhs=np.r_[-e,-1e-8*(q-initial)/np.sqrt(damping+1e-8)]
            lower=np.maximum(lo-q,-.2);upper=np.minimum(hi-q,.2)
            free=upper-lower>1e-12;dq=lower.copy()
            if np.any(free):dq[free]=lsq_linear(A[:,free],rhs-A[:,~free]@dq[~free],bounds=(lower[free],upper[free]),method='bvls',tol=1e-10).x
            cost=e@e; accepted=False
            for alpha in [1.,.5,.25,.125,.0625]:
                trial=np.clip(q+alpha*dq,lo,hi)
                er=self.weight*pin.log6(self.fk(trial).inverse()*target).vector
                if er@er < cost-1e-16:
                    q=trial;accepted=True;break
            if not accepted: break
        pose=self.fk(q)
        pe=float(np.linalg.norm(pose.translation-target.translation))
        oe=float(np.linalg.norm(pin.log3(pose.rotation.T@target.rotation)))
        return IKResult(q,pe,oe,pe<position_tol and oe<orientation_tol,it+1,time.perf_counter()-start)

    def reset(self,q):
        self.q_previous=np.asarray(q,dtype=float).copy()
        self.velocity=np.zeros_like(self.q_previous)

    def solve_point(self,target,seed,restarts=16):
        """Global point queries may restart; track() never calls this method."""
        from scipy.optimize import least_squares
        started=time.perf_counter()
        target=target if isinstance(target,pin.SE3) else pin.SE3(target)
        best=self.solve(target,seed,max_iter=150)
        if best.converged:return best
        def fun(q):return self.weight*pin.log6(self.fk(q).inverse()*target).vector
        def jac(q):
            delta=self.fk(q).inverse()*target
            J=pin.computeFrameJacobian(self.model,self.data,q,self.frame,pin.ReferenceFrame.LOCAL)
            return self.weight[:,None]*(-pin.Jlog6(delta.inverse())@J)
        rng=np.random.default_rng(531)
        seeds=[best.q,(self.lower+self.upper)/2]+list(rng.uniform(self.lower,self.upper,(restarts,6)))
        for initial in seeds:
            result=least_squares(fun,np.clip(initial,self.lower+1e-10,self.upper-1e-10),jac=jac,bounds=(self.lower,self.upper),max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            pose=self.fk(result.x);pe=float(np.linalg.norm(pose.translation-target.translation));oe=float(np.linalg.norm(pin.log3(pose.rotation.T@target.rotation)))
            candidate=IKResult(result.x,pe,oe,pe<1e-4 and oe<1e-3,result.nfev,time.perf_counter()-started)
            if pe+.3*oe<best.position_error+.3*best.orientation_error:best=candidate
            if candidate.converged:best=candidate;break
        best.seconds=time.perf_counter()-started
        return best

    def track(self,target,dt=.02,vmax=1.2,amax=6.):
        if self.q_previous is None: raise RuntimeError('Call reset(q) before trajectory tracking')
        prev=self.q_previous
        # Braking envelope prevents arriving at a joint limit with nonzero velocity.
        v_lo=np.maximum(-vmax,self.velocity-amax*dt)
        v_hi=np.minimum(vmax,self.velocity+amax*dt)
        v_lo=np.maximum(v_lo,-(np.sqrt((amax*dt)**2+2*amax*np.maximum(prev-self.lower,0))-amax*dt))
        v_hi=np.minimum(v_hi,(np.sqrt((amax*dt)**2+2*amax*np.maximum(self.upper-prev,0))-amax*dt))
        lo=np.maximum(self.lower,prev+v_lo*dt);hi=np.minimum(self.upper,prev+v_hi*dt)
        if np.any(lo>hi+1e-10): raise RuntimeError('Infeasible continuity bounds; trajectory must stop')
        result=self.solve(target,prev,(lo,hi),max_iter=40)
        self.velocity=(result.q-prev)/dt
        self.q_previous=result.q.copy()
        return result
