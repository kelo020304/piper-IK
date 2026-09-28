"""Optional official SDK bridge. Constructing this wrapper opens no CAN port."""
import time
import numpy as np

class HardwareLocked(PermissionError):pass

class MotorEnableLost(RuntimeError):
    def __init__(self,status):
        self.motor_enable_status=[bool(v) for v in status]
        detail=' / '.join(f'J{i+1}' for i,v in enumerate(status) if not v) if len(status)==6 else f'反馈数量异常：{len(status)}/6'
        super().__init__(f'电机使能丢失（SDK 反馈：{detail} 未使能），停止发送')

class MotorEnableTimeout(MotorEnableLost):
    def __init__(self,status):
        super().__init__(status)
        detail=' / '.join(f'J{i+1}' for i,v in enumerate(status) if not v) if len(status)==6 else '反馈不完整'
        self.args=(f'使能超时（2 秒）：{detail} 未确认使能；已停止，尚未回零',)

class PiperHardware:
    def __init__(self,config,authorized=False,sdk_factory=None):
        self.config=config;self.authorized=bool(authorized and config.get('hardware_enabled'))
        self.factory=sdk_factory;self.sdk=None;self.armed=False;self.estopped=False
        self.offset=np.asarray(config['joint_zero_offsets_rad'],dtype=float)
        self.last_feedback_stamp=None;self.last_feedback_time=0.
        self.last_command=None;self.last_send_time=None;self.velocity_limits=None
    def _allow(self):
        if not self.authorized:raise HardwareLocked('当前禁止真机连接和发送；硬件接口已锁定')
    def connect(self):
        self._allow()
        if self.sdk is not None:return
        factory=self.factory
        if factory is None:
            from piper_sdk import C_PiperInterface_V2
            factory=C_PiperInterface_V2
        # This is reachable only after both server-side authorization gates.
        self.sdk=factory(self.config['can_interface'],start_sdk_joint_limit=True)
        try:self.sdk.ConnectPort(piper_init=False)
        except Exception:
            try:self.sdk.DisconnectPort()
            finally:self.sdk=None
            raise
        self.armed=False;self.last_feedback_stamp=None;self.last_feedback_time=0.
    def feedback(self):
        if self.sdk is None:raise RuntimeError('机械臂未连接')
        msg=self.sdk.GetArmJointMsgs();stamp=float(msg.time_stamp)
        q=np.deg2rad(np.array([getattr(msg.joint_state,f'joint_{i}') for i in range(1,7)])/1000)-self.offset
        if not np.isfinite(q).all():raise RuntimeError('Invalid robot feedback')
        now=time.monotonic()
        if stamp>0 and stamp!=self.last_feedback_stamp:self.last_feedback_stamp=stamp;self.last_feedback_time=now
        if self.last_feedback_time==0 or now-self.last_feedback_time>.25:raise RuntimeError('关节反馈失效，禁止发送')
        return q
    def read_velocity_limits(self):
        """Read fresh SDK limits once per explicit enable; never change drive settings."""
        limits=[];self.velocity_limits=None
        for joint in range(1,7):
            previous=float(self.sdk.GetCurrentMotorAngleLimitMaxVel().time_stamp)
            self.sdk.SearchMotorMaxAngleSpdAccLimit(joint,0x01)
            deadline=time.monotonic()+.5
            while time.monotonic()<deadline:
                msg=self.sdk.GetCurrentMotorAngleLimitMaxVel()
                stamp=float(msg.time_stamp);motor=msg.current_motor_angle_limit_max_vel
                if stamp>0 and stamp!=previous and motor.motor_num==joint:
                    speed=float(motor.max_joint_spd)*.001  # SDK 0x473: 0.001 rad/s.
                    if not np.isfinite(speed) or speed<=0:raise RuntimeError(f'J{joint} SDK 最大速度无效，尚未使能')
                    limits.append(speed);break
                time.sleep(.01)
            else:raise RuntimeError(f'J{joint} SDK 最大速度读取超时，尚未使能')
        self.velocity_limits=np.array(limits)
        return self.velocity_limits.copy()
    def command_velocity_limits(self):
        if self.velocity_limits is None:raise RuntimeError('SDK 最大速度尚未读取')
        override=self.config.get('max_joint_velocity_rad_s')
        return self.velocity_limits.copy() if override is None else np.minimum(self.velocity_limits,override)
    def arm(self):
        self._allow()
        if self.sdk is None:raise RuntimeError('请先连接')
        if self.estopped:raise RuntimeError('急停已锁存，需要显式复位')
        q=self.feedback()
        self.read_velocity_limits()
        try:
            # The controller's software stop survives this Python process.
            # Drive enable alone does not resume it. Only an explicit arm
            # request reaches here; connect/startup remain read-only.
            self.sdk.MotionCtrl_1(0x02,0,0)
            deadline=time.monotonic()+2.;next_request=0.;status=[]
            while time.monotonic()<deadline:
                # Official EnablePiper examples repeat the request until
                # feedback confirms it. Bound retries to this user action.
                now=time.monotonic()
                if now>=next_request:
                    self.sdk.EnableArm(7);next_request=now+.1
                status=self.sdk.GetArmEnableStatus()
                if len(status)==6 and all(status):break
                self.feedback()
                time.sleep(.02)
            else:raise MotorEnableTimeout(status)
            q=self.feedback()
        except Exception as exc:
            exc.hardware_diagnostics=self.diagnostics()
            self.halt();raise
        self.last_command=q.copy();self.last_send_time=time.monotonic();self.armed=True
        return q
    def check_enabled(self):
        status=self.sdk.GetArmEnableStatus()
        if len(status)!=6 or not all(status):
            exc=MotorEnableLost(status);exc.hardware_diagnostics=self.diagnostics();raise exc
    def diagnostics(self):
        """Copy already cached SDK feedback before sending stop; no CAN requests."""
        result={}
        if hasattr(self.sdk,'GetArmStatus'):
            try:
                msg=self.sdk.GetArmStatus();status=msg.arm_status
                result['controller']={'time_stamp':float(msg.time_stamp),**{k:int(getattr(status,k)) for k in ('ctrl_mode','arm_status','mode_feed','motion_status')}}
            except Exception as exc:result['controller_read_error']=str(exc)
        if hasattr(self.sdk,'GetArmLowSpdInfoMsgs'):
            try:
                msg=self.sdk.GetArmLowSpdInfoMsgs();motors=[]
                flags=('voltage_too_low','motor_overheating','driver_overcurrent','driver_overheating','collision_status','driver_error_status','driver_enable_status','stall_status')
                for i in range(1,7):
                    m=getattr(msg,f'motor_{i}')
                    motors.append({'joint':i,'voltage_v':float(m.vol)/10,**{k:bool(getattr(m.foc_status,k)) for k in flags}})
                result['motors']={'time_stamp':float(msg.time_stamp),'values':motors}
            except Exception as exc:result['motor_read_error']=str(exc)
        return result
    def send(self,q,dt):
        self._allow()
        if not self.armed or self.estopped:raise RuntimeError('机械臂未使能或处于急停')
        self.check_enabled()
        self.feedback();q=np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Invalid joint command')
        if not np.isfinite(dt) or not 0<dt<=.1:raise ValueError('Invalid command timestep')
        if self.last_command is not None and np.any(abs(q-self.last_command)>self.command_velocity_limits()*dt+1e-6):raise RuntimeError('Joint command rate limit exceeded')
        units=np.rint(np.rad2deg(q+self.offset)*1000).astype(int)
        self.sdk.MotionCtrl_2(0x01,0x01,int(self.config['speed_percent']),0x00)
        self.sdk.JointCtrl(*map(int,units));self.last_command=q.copy();self.last_send_time=time.monotonic()
    def halt(self):
        self._allow()
        try:
            if self.sdk is not None:self.sdk.MotionCtrl_1(0x01,0,0)
        except Exception as exc:
            raise RuntimeError('急停发送失败，请使用物理急停：'+str(exc)) from exc
        finally:self.armed=False;self.estopped=True
    def clear_stop(self):
        self._allow()
        if self.sdk is None:raise RuntimeError('未连接')
        self.sdk.MotionCtrl_1(0x02,0,0);self.estopped=False;self.armed=False
    def disable(self):
        self._allow()
        if self.sdk is None:raise RuntimeError('未连接')
        self.halt()
        self.sdk.DisableArm(7)
        deadline=time.monotonic()+2.
        while time.monotonic()<deadline:
            status=self.sdk.GetArmEnableStatus()
            if len(status)==6 and not any(status):return
            time.sleep(.02)
        raise RuntimeError('失能未确认，请检查机械臂状态')
    def disconnect(self):
        self._allow()
        if self.sdk is not None:
            if self.armed:self.halt()
            self.sdk.DisconnectPort();self.sdk=None;self.velocity_limits=None
        self.armed=False
