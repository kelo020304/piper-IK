"""Local pose conditioning and jerk-limited joint trajectory generation."""
import numpy as np
from scipy.spatial.transform import Rotation
from ruckig import Ruckig, InputParameter, Trajectory, Result, ControlInterface


class PoseReference:
    def __init__(self,pose,tau=.06,linear_speed=.04,angular_speed=.5):
        self.tau=tau;self.linear_speed=linear_speed;self.angular_speed=angular_speed
        self.reset(pose)
    def reset(self,pose):self.stages=[pose.copy() for _ in range(3)]
    def step(self,target,dt):
        alpha=-np.expm1(-dt/self.tau);source=target
        for stage in self.stages:
            dp=alpha*(source[:3,3]-stage[:3,3])
            length=np.linalg.norm(dp)
            if self.linear_speed is not None and length>self.linear_speed*dt:dp*=self.linear_speed*dt/length
            rv=alpha*Rotation.from_matrix(source[:3,:3]@stage[:3,:3].T).as_rotvec()
            angle=np.linalg.norm(rv)
            if self.angular_speed is not None and angle>self.angular_speed*dt:rv*=self.angular_speed*dt/angle
            stage[:3,3]+=dp
            stage[:3,:3]=Rotation.from_rotvec(rv).as_matrix()@stage[:3,:3]
            source=stage
        return self.stages[-1].copy()


class JointTrajectory:
    def __init__(self,lo,hi,q,vmax,amax,jmax=10.):
        self.lo=np.array(lo);self.hi=np.array(hi);self.q=np.array(q).copy()
        self.v=np.zeros_like(self.q);self.velocity=self.v.copy();self.a=self.v.copy()
        self.vmax=np.broadcast_to(np.asarray(vmax,dtype=float),self.q.shape).copy();self.amax=amax;self.jmax=jmax
        if not np.isfinite(self.vmax).all() or np.any(self.vmax<=0):raise ValueError("Invalid joint velocity limits")
        self.generator=Ruckig(len(q));self.previous_goal=self.q.copy();self.boundary_braking=False
    def apply(self,proposal,dt,vmax=None,stop=False):
        if not np.isfinite(dt) or not 0<dt<=.1:raise ValueError('Invalid trajectory timestep')
        proposal=np.asarray(proposal,dtype=float)
        if proposal.shape!=self.q.shape or not np.isfinite(proposal).all():raise ValueError('Invalid joint trajectory target')
        n=len(self.q);inp=InputParameter(n)
        inp.current_position=self.q;inp.current_velocity=self.velocity;inp.current_acceleration=self.a
        inp.target_position=np.clip(proposal,self.lo,self.hi)
        inp.target_velocity=[0.]*n;inp.target_acceleration=[0.]*n
        inp.max_velocity=self.vmax if vmax is None else np.minimum(vmax,self.vmax)
        inp.max_acceleration=[self.amax]*n;inp.max_jerk=[self.jmax]*n
        # Avoid zero-duration multi-axis synchronization at floating-point
        # residuals while preserving velocity/acceleration continuity.
        inp.minimum_duration=dt
        if stop:inp.control_interface=ControlInterface.Velocity
        def plan():
            trajectory=Trajectory(n)
            result=self.generator.calculate(inp,trajectory)
            if result not in (Result.Working,Result.Finished):raise RuntimeError(f'轨迹生成失败：{result}')
            valid=all(ext.min>=lo-1e-8 and ext.max<=hi+1e-8 for ext,lo,hi in zip(trajectory.position_extrema,self.lo,self.hi))
            return trajectory,valid
        trajectory,valid=plan();self.boundary_braking=not valid
        if not valid:
            # Finish the last feasible braking trajectory rather than clipping
            # a position at the boundary and discontinuously resetting velocity.
            inp.control_interface=ControlInterface.Position;inp.target_position=self.previous_goal
            trajectory,valid=plan()
            if not valid:raise RuntimeError('当前运动状态无法生成限位内的连续轨迹')
        q,velocity,acceleration=map(np.asarray,trajectory.at_time(dt))
        self.previous_goal=np.array(inp.target_position)
        self.v=(q-self.q)/dt;self.q=q.copy();self.velocity=velocity.copy();self.a=acceleration.copy()
        return self.q.copy()
