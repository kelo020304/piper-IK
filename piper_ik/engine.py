"""Bounded EEF keyboard teleoperation, one controller lease, no automatic hardware IO."""
from pathlib import Path
import logging,threading,time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .pin_solver import PinIK
from .policy import Policy
from .hardware import PiperHardware,HardwareLocked
ROOT=Path(__file__).resolve().parents[1]
KEYS={'w':('p',0,1),'s':('p',0,-1),'a':('p',1,1),'d':('p',1,-1),'ArrowUp':('p',2,1),'ArrowDown':('p',2,-1),'q':('r',2,1),'e':('r',2,-1),'z':('r',1,1),'c':('r',1,-1),'r':('r',0,1),'f':('r',0,-1)}

# Compatibility export for offline analysis scripts.
from .trajectory import JointTrajectory as Limiter, PoseReference

class Engine:
    def __init__(self,config,authorized=False,sdk_factory=None):
        self.config=config;self.lock=threading.RLock();self.simulator=None;self.sim_feedback=None;self.simulator_ok=False;self.simulator_ms=0.
        self.model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/models/piper.xml'));self.data=mujoco.MjData(self.model)
        self.lo,self.hi=self.model.jnt_range.T;self.tip=self.model.body('tool0').id
        self.pin=PinIK(ROOT/'assets/models/piper.urdf');self.rl=Policy(ROOT/'assets/weights/ppo_precision.pt',self.lo,self.hi)
        self.tcp=np.eye(4);self.tcp[:3,3]=config['tcp_xyz_m'];self.tcp[:3,:3]=Rotation.from_euler('xyz',config['tcp_rpy_deg'],degrees=True).as_matrix()
        self.hardware=PiperHardware(config,authorized,sdk_factory)
        self.q=np.asarray(config['initial_q'],dtype=float);self.limiter=self.new_limiter(self.q)
        if self.q.shape!=(6,) or not np.isfinite(self.q).all() or np.any(self.q<self.lo) or np.any(self.q>self.hi):raise ValueError('Invalid initial joints')
        if (config.get('max_joint_velocity_rad_s') is not None and not 0<config['max_joint_velocity_rad_s']<=.5) or not 0<config['max_joint_acceleration_rad_s2']<=3:raise ValueError('Invalid teleoperation limits')
        if not 1<=config['speed_percent']<=20:raise ValueError('Hardware speed must be 1–20 percent')
        if not 0<config.get('home_velocity_rad_s',.08)<=.5:raise ValueError('Invalid home velocity')
        for key,default,maximum in [('max_joint_jerk_rad_s3',10.,100.),('target_filter_tau_s',.06,1.),('target_linear_speed_m_s',.04,.25),('target_angular_speed_rad_s',.5,2.)]:
            if config.get(key,default) is None and key in ('target_linear_speed_m_s','target_angular_speed_rad_s'):continue
            if not 0<config.get(key,default)<=maximum:raise ValueError('Invalid trajectory parameter: '+key)
        self.measured_q=self.q.copy()
        self.reference=PoseReference(self.pose(self.q),config.get('target_filter_tau_s',.06),config.get('target_linear_speed_m_s',.04),config.get('target_angular_speed_rad_s',.5))
        self.ik_seed=self.q.copy();self.reference_pose=self.pose(self.q)
        self.mode='pin';self.target=self.pose(self.q);self.paused=True;self.owner=None;self.last_heartbeat=0.;self.last_nudge=-1e10
        self.execution='hardware' if self.hardware.authorized else 'simulation';self.joint_goal=None;self.motion_started=0.;self.feedback_valid=False;self.message='点击接管键盘，在仿真中开始';self.fault=None
        self.last_fault=None;self.safe_q=None;self.motion_kind=None;self.disable_hold_since=None;self.owner_kind=None;self.previous_measured=None;self.previous_measured_time=None
        self.step_mm=float(config['position_step_mm']);self.step_deg=float(config['rotation_step_deg']);self.solver_ms=0.;self.frames=0
    def attach_simulator(self,simulator):
        with self.lock:
            if self.hardware.authorized and simulator.backend!='mujoco':raise HardwareLocked('真机同步仅支持 MuJoCo')
            self.simulator=simulator
            self.sync_simulator(self.q)
            actual=np.array(self.sim_feedback['flange'])@self.tcp
            if np.linalg.norm(actual[:3,3]-self.target[:3,3])>1e-4:raise RuntimeError('仿真模型初始位姿与 IK 模型不一致')
            self.target=actual;self.message='真机会话待连接，模型为占位姿态' if self.hardware.authorized else f'{simulator.backend} 已连接，点击接管键盘'
    def sync_simulator(self,q):
        start=time.perf_counter();feedback=self.simulator.step(q,self.target,self.tcp)
        angles=np.asarray(feedback['q']);T=np.asarray(feedback['flange'])
        if angles.shape!=(6,) or T.shape!=(4,4) or not np.isfinite(angles).all() or not np.isfinite(T).all():raise RuntimeError('无效仿真反馈')
        if np.max(abs(angles-q))>1e-4:raise RuntimeError('仿真关节反馈与命令不一致')
        self.sim_feedback=feedback;self.simulator_ok=True;self.simulator_ms=(time.perf_counter()-start)*1000
    def actual_pose(self):
        if self.execution=='simulation' and self.sim_feedback is not None:return np.array(self.sim_feedback['flange'])@self.tcp
        return self.pose(self.measured_q if self.execution=='hardware' else self.q)
    def new_limiter(self,q):
        # Unarmed hardware uses model bounds only for its stationary placeholder.
        # Explicit arm reads drive bounds before a trajectory can be sent.
        vmax=self.pin.model.velocityLimit.copy()
        if self.hardware.velocity_limits is not None:vmax=self.hardware.command_velocity_limits()
        elif self.config.get('max_joint_velocity_rad_s') is not None:vmax=np.minimum(vmax,self.config['max_joint_velocity_rad_s'])
        return Limiter(self.lo,self.hi,q,vmax,self.config['max_joint_acceleration_rad_s2'],self.config.get('max_joint_jerk_rad_s3',10.))
    def reset_reference(self):
        self.reference_pose=self.pose(self.q);self.reference.reset(self.reference_pose);self.ik_seed=self.q.copy()
    def flange(self,q):
        self.data.qpos[:]=q;mujoco.mj_kinematics(self.model,self.data)
        T=np.eye(4);T[:3,3]=self.data.xpos[self.tip];T[:3,:3]=self.data.xmat[self.tip].reshape(3,3);return T
    def pose(self,q):return self.flange(q)@self.tcp
    def require_owner(self,client):
        if self.owner!=client:raise PermissionError('请先接管键盘')
    def control(self,client,msg):
        with self.lock:
            op=msg.get('type')
            if op=='takeover':
                self.stop('已接管到本页；如已停止，请清除故障后使能')
                self.owner=client;self.owner_kind=msg.get('source','other');self.last_heartbeat=time.monotonic()
                return
            if op=='claim':
                if self.joint_goal is not None:raise RuntimeError('自动运动中；先停止再接管')
                if self.owner is not None and self.owner!=client:raise PermissionError('其他控制端占用，请点击“接管到本页”')
                self.owner=client;self.owner_kind=msg.get('source','other');self.last_heartbeat=time.monotonic();self.paused=self.execution=='hardware' and not self.hardware.armed;self.fault=self.fault if self.execution=='hardware' else None;self.target=self.actual_pose();self.message='键盘已接管';return
            self.require_owner(client)
            if op=='heartbeat':self.last_heartbeat=time.monotonic();return
            if op in ['release','pause']:
                self.stop('已暂停');self.owner=None if op=='release' else client;return
            if op=='mode':
                if self.joint_goal is not None:raise RuntimeError('自动运动中不能切换 IK')
                mode=msg.get('mode')
                if mode not in ['pin','rl']:raise ValueError('Unknown IK mode')
                self.mode=mode;self.target=self.actual_pose();self.message='Pinocchio' if mode=='pin' else 'RL · 已载入训练权重';return
            if op=='steps':
                mm=float(msg.get('mm',self.step_mm));deg=float(msg.get('deg',self.step_deg))
                if not np.isfinite([mm,deg]).all() or not .1<=mm<=5 or not .1<=deg<=5:raise ValueError('步长限于 0.1–5 mm / 0.1–5°')
                self.step_mm=mm;self.step_deg=deg;return
            if op=='reset':
                if self.execution!='simulation':raise HardwareLocked('真机不支持仿真复位')
                self.q=np.array(self.config['initial_q'],dtype=float);self.limiter=self.new_limiter(self.q);self.target=self.pose(self.q);self.paused=True;self.message='仿真已复位'
                self.reset_reference()
                if self.simulator:self.sync_simulator(self.q)
                return
            if op=='pose_target':
                if self.paused or self.joint_goal is not None:raise RuntimeError('请先接管并结束自动运动')
                if self.execution=='hardware' and not self.hardware.armed:raise HardwareLocked('真机未使能')
                if msg.get('frame')!='base':raise ValueError('末端目标必须使用 base 坐标系')
                position=np.asarray(msg.get('position_m'),dtype=float);quat=np.asarray(msg.get('quaternion_xyzw'),dtype=float)
                if position.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(position).all() or not np.isfinite(quat).all():raise ValueError('末端目标格式无效')
                if np.linalg.norm(position)>.85 or abs(np.linalg.norm(quat)-1)>.001:raise ValueError('末端目标位置或单位四元数无效')
                target=np.eye(4);target[:3,3]=position;target[:3,:3]=Rotation.from_quat(quat).as_matrix()
                self.target=target;self.message='末端目标已更新';return
            if op=='nudge':
                if self.joint_goal is not None:raise RuntimeError('自动运动中不能微调')
                if self.paused:raise ValueError('当前已暂停，重新接管键盘后移动')
                if self.execution=='hardware' and not self.hardware.armed:raise HardwareLocked('真机未使能')
                key=msg.get('key')
                if key not in KEYS:raise ValueError('Unknown key')
                now=time.monotonic()
                if now-self.last_nudge<.09:return
                self.last_nudge=now;kind,axis,sign=KEYS[key];candidate=self.target.copy()
                if kind=='p':candidate[axis,3]+=sign*self.step_mm/1000
                else:
                    rv=np.zeros(3);rv[axis]=sign*np.deg2rad(self.step_deg);candidate[:3,:3]=Rotation.from_rotvec(rv).as_matrix()@candidate[:3,:3]
                if np.linalg.norm(candidate[:3,3])>.85:raise ValueError('超出目标范围')
                self.target=candidate;self.message=f'{key} · 微调';return
            raise ValueError('Unknown command')
    def cancel_joint_motion(self):
        self.joint_goal=None;self.motion_kind=None;self.disable_hold_since=None
    def stop(self,message):
        self.cancel_joint_motion();self.paused=True;self.target=self.actual_pose();self.message=message
        if self.execution=='hardware' and self.hardware.armed:
            try:self.hardware.halt()
            except Exception as exc:self.fault=str(exc);self.message=str(exc)
    def hardware_action(self,client,action):
        with self.lock:
            self.hardware._allow();self.require_owner(client)
            if action=='connect':
                if self.hardware.sdk is not None:raise RuntimeError('已连接')
                self.hardware.connect();self.execution='hardware';self.feedback_valid=False;self.paused=True;self.message='已连接，等待反馈并手动使能'
            elif action=='arm':
                if self.hardware.armed:raise RuntimeError('已使能')
                if self.simulator and not self.simulator_ok:raise RuntimeError('MuJoCo 未就绪')
                measured=self.hardware.feedback()
                if np.any(measured<self.lo) or np.any(measured>self.hi):raise RuntimeError('实测角超出模型限位，禁止使能')
                try:q=self.hardware.arm()
                except Exception as exc:
                    # Action errors used to disappear after the transient WS
                    # notification, leaving the previous "cleared" message.
                    self.cancel_joint_motion();self.paused=True
                    self.fault=f'使能失败：{exc}';self.message=self.fault
                    self.last_fault={'reason':self.fault,'action':'arm','mode':self.mode,'time':time.time(),
                        'motor_enable_status':getattr(exc,'motor_enable_status',None),
                        'hardware_diagnostics':getattr(exc,'hardware_diagnostics',{})}
                    self.last_heartbeat=time.monotonic()
                    logging.getLogger(__name__).warning('%s | %s',self.fault,self.last_fault)
                    raise
                if np.any(q<self.lo) or np.any(q>self.hi):self.hardware.halt();raise RuntimeError('实测角超出模型限位')
                if self.safe_q is None:self.safe_q=q.copy()
                self.q=q;self.measured_q=q.copy();self.limiter=self.new_limiter(q);self.target=self.pose(q);self.paused=False;self.message='真机已使能';self.fault=None;self.last_heartbeat=time.monotonic()
                self.reset_reference()
            elif action=='home':
                if not self.hardware.armed or self.paused:raise RuntimeError('请先使能')
                if self.joint_goal is not None:raise RuntimeError('已在回零')
                goal=-self.hardware.offset.copy()
                if np.any(goal<self.lo) or np.any(goal>self.hi):raise RuntimeError('零角度超出模型限位')
                self.motion_kind='home';self.disable_hold_since=None;self.joint_goal=goal;self.target=self.pose(goal);self.motion_started=time.monotonic();self.message='缓速回零中'
            elif action=='disable':
                if not self.hardware.armed or self.paused:raise RuntimeError('先恢复使能，再返回安全位置并失能')
                if self.safe_q is None:raise RuntimeError('尚未记录首次使能位置，禁止自动失能')
                if self.motion_kind in ('return_safe','disable_wait'):raise RuntimeError('已在返回失能流程中')
                self.joint_goal=self.safe_q.copy();self.motion_kind='return_safe';self.disable_hold_since=None
                self.target=self.pose(self.joint_goal);self.motion_started=time.monotonic();self.message='返回首次使能位置，到位保持 3 秒后失能'
            elif action=='stop':
                self.cancel_joint_motion();self.hardware.halt();self.paused=True;self.message='真机急停已锁存'
            elif action=='clear':
                if self.hardware.armed:raise RuntimeError('先停止，再清除故障')
                # Require a valid current state; clearing never enables or moves.
                q=self.hardware.feedback()
                if np.any(q<self.lo) or np.any(q>self.hi):raise RuntimeError('实测角超出模型限位，无法清除故障')
                # A previous process may have stopped the controller even
                # though this wrapper's local estopped flag starts False.
                self.hardware.clear_stop()
                self.cancel_joint_motion();self.q=q.copy();self.measured_q=q.copy();self.limiter=self.new_limiter(q)
                self.target=self.pose(q);self.feedback_valid=True;self.paused=True;self.fault=None
                self.reset_reference()
                if self.last_fault:self.last_fault={**self.last_fault,'cleared_at':time.time()}
                self.message='故障已清除，等待使能';self.last_heartbeat=time.monotonic()
            elif action=='disconnect':
                self.cancel_joint_motion();self.hardware.disconnect();self.safe_q=None;self.previous_measured=None;self.previous_measured_time=None;self.feedback_valid=False;self.paused=True;self.message='真机已断开';self.limiter=self.new_limiter(self.q);self.target=self.pose(self.q)
            else:raise ValueError('Unknown hardware action')
    def tick(self,dt=.02):
        with self.lock:
            start=time.perf_counter();now=time.monotonic()
            if self.owner and time.monotonic()-self.last_heartbeat>self.config['heartbeat_timeout_s']:
                self.stop('控制连接超时，已停止');self.owner=None
            try:
                measured=None
                if self.execution=='hardware':
                    if self.hardware.sdk is None:return
                    self.feedback_valid=False
                    try:measured=self.hardware.feedback()
                    except Exception as exc:
                        if self.hardware.armed or self.joint_goal is not None:raise
                        self.paused=True;self.message='等待有效关节反馈：'+str(exc)
                        return
                    if np.any(measured<self.lo-.001) or np.any(measured>self.hi+.001):raise RuntimeError('实测关节超出模型限位')
                    self.feedback_valid=True
                    if self.hardware.armed:self.hardware.check_enabled()
                    self.measured_q=measured.copy()
                    measured_speed=float('inf') if self.previous_measured is None else np.max(abs(measured-self.previous_measured))/max(now-self.previous_measured_time,1e-6)
                    self.previous_measured=measured.copy();self.previous_measured_time=now
                    if not self.hardware.armed:
                        if self.joint_goal is not None:raise RuntimeError('使能丢失，自动运动已取消')
                        self.q=measured;self.limiter=self.new_limiter(self.q);self.target=self.pose(self.q);self.paused=True
                if self.paused or self.joint_goal is not None:
                    self.reference.reset(self.pose(self.q));self.ik_seed=self.q.copy();self.reference_pose=self.pose(self.q)
                if self.paused:proposal=self.q.copy()
                elif self.joint_goal is not None:
                    if time.monotonic()-self.motion_started>90:raise RuntimeError('自动运动超时')
                    proposal=self.joint_goal.copy()
                    arrived=(np.max(abs(measured-self.joint_goal))<.0035 and np.max(abs(self.q-self.joint_goal))<.0035 and np.max(abs(self.limiter.v))<.001 and measured_speed<.01)
                    if self.motion_kind=='home':
                        if arrived:
                            self.cancel_joint_motion();self.target=self.pose(self.q);self.message='回零完成，保持当前位置'
                    elif arrived:
                        if self.disable_hold_since is None:self.disable_hold_since=now
                        self.motion_kind='disable_wait'
                        remaining=max(0.,3.-(now-self.disable_hold_since))
                        self.message=f'安全位置保持中 · {remaining:.1f} 秒后失能'
                        if remaining<=0:
                            self.hardware.disable();self.cancel_joint_motion();self.paused=True
                            self.q=measured.copy();self.limiter=self.new_limiter(self.q);proposal=self.q.copy()
                            self.target=self.pose(self.q);self.message='已返回首次使能位置并保持 3 秒，已失能'
                    else:
                        self.disable_hold_since=None;self.motion_kind='return_safe';self.message='返回首次使能位置'
                else:
                    self.reference_pose=self.reference.step(self.target,dt)
                    goal_flange=self.reference_pose@np.linalg.inv(self.tcp)
                    # Keep IK branch continuity separately from measured drive lag.
                    seed=self.ik_seed.copy()
                    if self.mode=='pin':proposal=self.pin.solve(goal_flange,seed,max_iter=40).q
                    else:proposal=self.rl.solve(seed,goal_flange,self.flange)
                    self.ik_seed=proposal.copy()
                nextq=self.limiter.apply(proposal,dt,vmax=self.config.get('home_velocity_rad_s',.08) if self.joint_goal is not None else None,stop=self.paused)
                if self.execution=='hardware' and self.hardware.armed:self.hardware.send(nextq,dt)
                self.q=nextq
                if self.execution=='simulation':
                    self.measured_q=self.q.copy()
                    if self.simulator:self.sync_simulator(self.q)
                elif self.simulator:self.sync_simulator(self.measured_q)
                if self.paused:self.target=self.pose(self.q)
                self.solver_ms=(time.perf_counter()-start)*1000;self.frames+=1
            except Exception as exc:
                self.last_fault={'reason':str(exc),'motion_kind':self.motion_kind,'mode':self.mode,'measured_q':self.measured_q.tolist(),'command_q':self.q.tolist(),'tracking_error':getattr(exc,'tracking_error',None),'time':time.time()}
                if hasattr(exc,'motor_enable_status'):
                    self.last_fault['motor_enable_status']=exc.motor_enable_status
                if hasattr(exc,'hardware_diagnostics'):
                    self.last_fault['hardware_diagnostics']=exc.hardware_diagnostics
                self.cancel_joint_motion();self.fault=str(exc);self.paused=True;self.message='已停止：'+str(exc)
                if self.simulator:self.simulator_ok=False
                if self.execution=='hardware' and self.hardware.armed:
                    try:self.hardware.halt()
                    except Exception as halt_error:self.fault=str(halt_error);self.message=str(halt_error)
    def snapshot(self):
        with self.lock:
            shown_q=self.measured_q if self.execution=='hardware' or self.sim_feedback is not None else self.q
            T=self.pose(shown_q);rv=Rotation.from_matrix(T[:3,:3]).as_euler('xyz',degrees=True)
            links=[]
            for name in ['base_link']+[f'link{i}' for i in range(1,7)]:
                b=self.model.body(name).id;quat=self.data.xquat[b];links.append({'name':name,'position':self.data.xpos[b].tolist(),'quaternion':[float(x) for x in quat[[1,2,3,0]]]})
            if self.execution=='simulation' and self.sim_feedback is not None:
                T=np.array(self.sim_feedback['flange'])@self.tcp;links=self.sim_feedback['links']
            command_T=self.pose(self.q)
            command_pe=float(np.linalg.norm(command_T[:3,3]-self.target[:3,3])*1000)
            command_oe=float(np.rad2deg(Rotation.from_matrix(command_T[:3,:3].T@self.target[:3,:3]).magnitude()))
            near_limits=(np.flatnonzero(np.minimum(self.q-self.lo,self.hi-self.q)<np.deg2rad(.5))+1).tolist()
            def pose_data(matrix):return {'position':matrix[:3,3].tolist(),'quaternion':Rotation.from_matrix(matrix[:3,:3]).as_quat().tolist(),'rpy':Rotation.from_matrix(matrix[:3,:3]).as_euler('xyz',degrees=True).tolist()}
            return {'joint_velocity_limits_rad_s':self.limiter.vmax.tolist(),'sdk_velocity_limits_rad_s':self.hardware.velocity_limits.tolist() if self.hardware.velocity_limits is not None else None,'reference':pose_data(self.reference_pose),'command_acceleration':self.limiter.a.tolist(),'boundary_braking':self.limiter.boundary_braking,'command_position_error_mm':command_pe,'command_orientation_error_deg':command_oe,'near_joint_limits':near_limits,'last_fault':self.last_fault,'owner_kind':self.owner_kind if self.owner else None,'safe_q':self.safe_q.tolist() if self.safe_q is not None else None,'motion_kind':self.motion_kind,'motion_busy':self.joint_goal is not None,'disable_remaining_s':max(0.,3.-(time.monotonic()-self.disable_hold_since)) if self.disable_hold_since is not None else None,'homing':self.motion_kind=='home','feedback_valid':self.feedback_valid,'feedback_age_ms':(time.monotonic()-self.hardware.last_feedback_time)*1000 if self.hardware.last_feedback_time else None,'simulator':self.simulator.backend if self.simulator else 'mujoco','simulator_connected':bool(self.simulator and self.simulator_ok and self.simulator.process.poll() is None),'simulator_frames':self.sim_feedback['frames'] if self.sim_feedback else 0,'simulator_ms':self.simulator_ms,'native_window':bool(self.simulator and not self.simulator.headless),'mode':self.mode,'execution':self.execution,'paused':self.paused,'owner':self.owner,'message':self.message,'fault':self.fault,'q':shown_q.tolist(),'command_q':self.q.tolist(),'velocity':self.limiter.v.tolist(),'links':links,'actual':pose_data(T),'target':pose_data(self.target),'position_error_mm':float(np.linalg.norm(T[:3,3]-self.target[:3,3])*1000),'orientation_error_deg':float(np.rad2deg(Rotation.from_matrix(T[:3,:3].T@self.target[:3,:3]).magnitude())),'solver_ms':self.solver_ms,'frames':self.frames,'step_mm':self.step_mm,'step_deg':self.step_deg,'hardware_allowed':self.hardware.authorized,'hardware_connected':self.hardware.sdk is not None,'hardware_armed':self.hardware.armed,'hardware_estopped':self.hardware.estopped,'tcp_xyz_m':self.config['tcp_xyz_m'],'policy_steps':self.rl.steps}
