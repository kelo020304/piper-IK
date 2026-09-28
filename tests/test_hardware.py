import json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from piper_ik.hardware import PiperHardware
ROOT=Path(__file__).resolve().parents[1]
class FakeSDK:
 def __init__(self,*a,**k):self.calls=[];self.units=np.rint(np.rad2deg([0,1,-1.2,.2,.55,0])*1000).astype(int);self.stale=False
 def ConnectPort(self,**kwargs):self.calls.append(('connect',kwargs))
 def GetArmJointMsgs(self):return SimpleNamespace(time_stamp=0 if self.stale else time.time(),joint_state=SimpleNamespace(**{f'joint_{i+1}':int(v) for i,v in enumerate(self.units)}))
 def GetCurrentMotorAngleLimitMaxVel(self):
  return getattr(self,'speed_reply',SimpleNamespace(time_stamp=0.,current_motor_angle_limit_max_vel=SimpleNamespace(motor_num=0,max_joint_spd=0)))
 def SearchMotorMaxAngleSpdAccLimit(self,joint,kind):
  assert kind==1
  stamp=self.GetCurrentMotorAngleLimitMaxVel().time_stamp+1
  self.speed_reply=SimpleNamespace(time_stamp=stamp,current_motor_angle_limit_max_vel=SimpleNamespace(motor_num=joint,max_joint_spd=[3000,3000,3000,3000,3000,2000][joint-1]))
 def GetArmEnableStatus(self):return [True]*6
 def EnableArm(self,n):self.calls.append(('enable',n))
 def MotionCtrl_2(self,*args):self.calls.append(('mode',args))
 def JointCtrl(self,*args):self.calls.append(('joints',args));self.units=np.array(args)
 def MotionCtrl_1(self,*args):self.calls.append(('stop',args))
 def DisconnectPort(self):self.calls.append(('disconnect',))

def test_sdk_units_feedback_enable_stop_and_no_automatic_motion():
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,authorized=True,sdk_factory=FakeSDK);assert h.sdk is None;h.connect();sdk=h.sdk
 assert [c[0] for c in sdk.calls]==['connect'];assert sdk.calls[0][1]=={'piper_init':False}
 q=h.arm();assert np.allclose(q,[0,1,-1.2,.2,.55,0],atol=1e-5)
 target=q.copy();target[0]+=.001;h.send(target,.02)
 assert sdk.calls[-1][0]=='joints';assert sdk.calls[-1][1][0]==round(np.rad2deg(.001)*1000)
 assert sdk.calls[-2]==('mode',(1,1,5,0))
 with pytest.raises(RuntimeError):h.send(target+.1,.02)
 h.halt();assert not h.armed and h.estopped
 with pytest.raises(RuntimeError):h.send(target,.02)
 h.clear_stop();assert not h.estopped and not h.armed
 h.disconnect();assert h.sdk is None

def test_stale_feedback_blocks_enable_and_send():
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,FakeSDK);h.connect();h.sdk.stale=True
 with pytest.raises(RuntimeError):h.arm()
 assert [c[0] for c in h.sdk.calls]==['connect']

def test_resume_send_failure_blocks_enable_and_latches_stop():
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,FakeSDK);h.connect();sdk=h.sdk
 def resume_fails(*args):
  if args[0]==2:raise OSError('resume failed')
  FakeSDK.MotionCtrl_1(sdk,*args)
 sdk.MotionCtrl_1=resume_fails
 with pytest.raises(OSError,match='resume failed'):h.arm()
 assert not h.armed and h.estopped
 assert sdk.calls==[('connect',{'piper_init':False}),('stop',(1,0,0))]

def test_explicit_stop_still_requires_clear_before_enable():
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,FakeSDK);h.connect();h.arm();h.halt();count=len(h.sdk.calls)
 with pytest.raises(RuntimeError,match='显式复位'):h.arm()
 assert len(h.sdk.calls)==count and not h.armed


def test_enable_retries_within_explicit_request_and_sends_no_joint_target(monkeypatch):
 clock=SimpleNamespace(now=100.)
 monkeypatch.setattr('piper_ik.hardware.time.monotonic',lambda:clock.now)
 monkeypatch.setattr('piper_ik.hardware.time.sleep',lambda dt:setattr(clock,'now',clock.now+dt))
 class DelayedSDK(FakeSDK):
  def GetArmEnableStatus(self):return [sum(c[0]=='enable' for c in self.calls)>=3]*6
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,DelayedSDK);h.connect();q=h.arm()
 assert h.armed and 100.2<=clock.now<102
 assert sum(c[0]=='enable' for c in h.sdk.calls)==3
 assert not any(c[0] in ('mode','joints') for c in h.sdk.calls)
 assert np.allclose(q,h.feedback())


def test_enable_timeout_persists_reason_and_pre_stop_diagnostics(monkeypatch):
 from piper_ik.engine import Engine
 clock=SimpleNamespace(now=100.)
 monkeypatch.setattr('piper_ik.hardware.time.monotonic',lambda:clock.now)
 monkeypatch.setattr('piper_ik.hardware.time.sleep',lambda dt:setattr(clock,'now',clock.now+dt))
 class RefusedSDK(FakeSDK):
  def GetArmEnableStatus(self):return [True,False,True,True,False,True]
  def GetArmStatus(self):
   stopped=any(c==('stop',(1,0,0)) for c in self.calls)
   return SimpleNamespace(time_stamp=123.,arm_status=SimpleNamespace(ctrl_mode=1,arm_status=1 if stopped else 6,mode_feed=1,motion_status=0))
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,RefusedSDK);e.control('t',{'type':'claim'});e.hardware_action('t','connect');e.hardware_action('t','clear')
 with pytest.raises(RuntimeError,match='J2 / J5'):e.hardware_action('t','arm')
 assert 102<=clock.now<102.1 and e.hardware.estopped and not e.hardware.armed
 assert e.safe_q is None and e.joint_goal is None
 assert e.fault==e.message and '使能失败' in e.fault
 assert e.last_fault['action']=='arm'
 assert e.last_fault['hardware_diagnostics']['controller']['arm_status']==6
 assert e.hardware.sdk.GetArmStatus().arm_status.arm_status==1
 requests=sum(c[0]=='enable' for c in e.hardware.sdk.calls)
 assert 2<requests<=20
 for _ in range(160):clock.now+=.02;e.last_heartbeat=clock.now;e.tick()
 state=e.snapshot()
 assert state['fault']==state['message'] and 'J2 / J5' in state['fault']
 assert not any(c[0]=='joints' for c in e.hardware.sdk.calls)
 assert sum(c[0]=='enable' for c in e.hardware.sdk.calls)==requests
 e.hardware_action('t','clear');assert e.fault is None and 'cleared_at' in e.last_fault


def test_enable_retry_aborts_when_joint_feedback_is_lost(monkeypatch):
 clock=SimpleNamespace(now=100.)
 monkeypatch.setattr('piper_ik.hardware.time.monotonic',lambda:clock.now)
 monkeypatch.setattr('piper_ik.hardware.time.sleep',lambda dt:setattr(clock,'now',clock.now+dt))
 class StaleDuringEnable(FakeSDK):
  def EnableArm(self,n):super().EnableArm(n);self.stale=True
  def GetArmEnableStatus(self):return [False]*6
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,StaleDuringEnable);h.connect()
 with pytest.raises(RuntimeError,match='反馈失效'):h.arm()
 assert clock.now<100.4 and h.estopped and not h.armed

def test_engine_future_hardware_path_only_uses_injected_fake():
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.control('test',{'type':'claim'});e.hardware_action('test','connect');sdk=e.hardware.sdk
 assert not any(x[0]=='enable' for x in sdk.calls)
 e.hardware_action('test','arm');e.control('test',{'type':'nudge','key':'w'})
 for _ in range(30):e.last_heartbeat=time.monotonic();e.tick()
 assert any(x[0]=='joints' for x in sdk.calls);assert e.fault is None
 assert np.allclose(e.snapshot()['q'],e.measured_q)
 e.last_heartbeat=time.monotonic()-1;e.tick();assert e.paused and e.hardware.estopped and not e.hardware.armed
 n=sum(x[0]=='joints' for x in sdk.calls)
 for _ in range(5):e.tick()
 assert sum(x[0]=='joints' for x in sdk.calls)==n

def test_sdk_send_failure_still_latches_stop():
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,FakeSDK);h.connect();h.arm()
 def broken(*args):raise OSError('CAN unavailable')
 h.sdk.MotionCtrl_1=broken
 with pytest.raises(RuntimeError,match='物理急停'):h.halt()
 assert h.estopped and not h.armed

@pytest.mark.parametrize('mode',['pin','rl'])
def test_hardware_feedback_mirror_home_and_ik(mode):
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.mode=mode
 class Mirror:
  backend='mujoco';headless=True;process=SimpleNamespace(poll=lambda:None)
  def step(self,q,target,tcp):
   self.q=q.copy()
   return dict(q=q.tolist(),flange=e.flange(q).tolist(),links=[],frames=1)
 mirror=Mirror();e.attach_simulator(mirror)
 for _ in range(3):e.tick()
 assert e.hardware.sdk is None and not e.feedback_valid
 e.control('test',{'type':'claim'})
 with pytest.raises(RuntimeError):e.hardware_action('test','home')
 e.hardware_action('test','connect');e.tick();e.hardware_action('test','arm')
 e.control('test',{'type':'nudge','key':'w'})
 for _ in range(60):
  before=e.hardware.feedback().copy();e.last_heartbeat=time.monotonic();e.tick()
  assert e.fault is None
  assert np.allclose(mirror.q,before) # mirror measurements, never next command
 assert e.snapshot()['position_error_mm']<.7
 e.hardware_action('test','home')
 with pytest.raises(RuntimeError):e.control('test',{'type':'nudge','key':'s'})
 with pytest.raises(RuntimeError):e.control('test',{'type':'mode','mode':'rl'})
 previous=e.q.copy();previous_v=e.limiter.v.copy()
 for _ in range(1800):
  e.last_heartbeat=time.monotonic();e.tick();assert e.fault is None
  assert np.all(abs(e.q-previous)<=e.limiter.vmax*.02+1e-8)
  assert max(abs(e.limiter.v-previous_v))<=config['max_joint_acceleration_rad_s2']*.02+1e-8
  previous=e.q.copy();previous_v=e.limiter.v.copy()
  if e.joint_goal is None:break
 else:raise AssertionError('home did not finish')
 assert max(abs(e.hardware.feedback()))<.0035
 q=e.q.copy()
 for _ in range(20):e.last_heartbeat=time.monotonic();e.tick()
 assert np.max(abs(e.q-q))<1e-4
 e.hardware_action('test','disconnect');assert e.execution=='hardware' and e.paused
 e.tick();assert e.hardware.sdk is None and not e.feedback_valid


def test_stale_or_lost_enable_stops_and_blocks_more_commands():
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 for failure in ('stale','disabled'):
  e=Engine(config,True,FakeSDK);e.control('t',{'type':'claim'});e.hardware_action('t','connect');e.hardware_action('t','arm');sdk=e.hardware.sdk
  if failure=='stale':sdk.stale=True;e.hardware.last_feedback_time=time.monotonic()-1
  else:sdk.GetArmEnableStatus=lambda:[False]*6
  n=len([x for x in sdk.calls if x[0]=='joints']);e.tick()
  assert not e.hardware.armed and e.hardware.estopped and e.fault
  assert len([x for x in sdk.calls if x[0]=='joints'])==n


def test_disable_is_explicit_and_verified():
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,FakeSDK);h.connect();h.arm();sdk=h.sdk
 def disable(n):sdk.calls.append(('disable',n));sdk.GetArmEnableStatus=lambda:[False]*6
 sdk.DisableArm=disable
 h.disable();assert not h.armed and h.estopped and sdk.calls[-1]==('disable',7)


def test_enable_loss_records_affected_motors_before_stop():
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.control('t',{'type':'claim'});e.hardware_action('t','connect');e.hardware_action('t','arm')
 sdk=e.hardware.sdk;sdk.GetArmEnableStatus=lambda:[True,False,True,False,True,True]
 count=sum(c[0]=='joints' for c in sdk.calls);e.tick()
 assert 'J2 / J4' in e.fault and 'J1' not in e.fault
 assert e.last_fault['motor_enable_status']==[True,False,True,False,True,True]
 assert not e.hardware.armed and e.hardware.estopped
 assert sum(c[0]=='joints' for c in sdk.calls)==count
 e.hardware_action('t','clear')
 assert e.last_fault['motor_enable_status']==[True,False,True,False,True,True]


def test_real_mujoco_process_mirrors_fake_hardware_for_both_solvers():
 from piper_ik.engine import Engine
 from piper_ik.simulator import SimulatorBridge
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);sim=SimulatorBridge('mujoco',headless=True,timeout=20)
 try:
  e.attach_simulator(sim);e.control('t',{'type':'claim'});e.hardware_action('t','connect');e.tick();e.hardware_action('t','arm')
  for mode in ('pin','rl'):
   e.control('t',{'type':'mode','mode':mode});e.last_nudge=-1e10;e.control('t',{'type':'nudge','key':'w'})
   for _ in range(50):
    before=e.hardware.feedback().copy();e.last_heartbeat=time.monotonic();e.tick()
    assert e.fault is None
    assert np.max(abs(np.array(e.sim_feedback['q'])-before))<1e-7
   assert e.snapshot()['position_error_mm']<.7
  e.hardware_action('t','home');e.hardware_action('t','stop')
  assert e.joint_goal is None and e.hardware.estopped
 finally:sim.close()


def test_authorized_server_still_does_not_connect_at_start_and_owner_disconnect_stops(tmp_path):
 from fastapi.testclient import TestClient
 from piper_ik.server import create_app
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 path=tmp_path/'config.json';path.write_text(json.dumps(config));calls=[]
 def factory(*a,**kw):
  sdk=FakeSDK();calls.append(sdk);return sdk
 app=create_app(path,authorized=True,sdk_factory=factory)
 with TestClient(app,base_url='http://127.0.0.1:8765') as client:
  assert client.get('/api/state').json()['execution']=='hardware';assert not calls
  assert client.get('/hardware-guide/').status_code==200
  with client.websocket_connect('ws://127.0.0.1:8765/ws',headers={'origin':'http://127.0.0.1:8765'}) as ws:
   assert ws.receive_json()['type']=='hello'
   for msg in ({'type':'claim'},{'type':'hardware','action':'connect'},{'type':'hardware','action':'arm'}):
    ws.send_json(msg)
    while True:
     response=ws.receive_json()
     assert response['type']!='error',response
     if response['type']=='ack':break
   assert app.state.engine.hardware.armed
  for _ in range(100):
   if not app.state.engine.hardware.armed:break
   time.sleep(.01)
  assert app.state.engine.hardware.estopped and not app.state.engine.hardware.armed
 assert calls[0].calls[-1][0]=='disconnect'


def test_enable_feedback_exception_latches_stop():
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,FakeSDK);h.connect()
 def broken():raise OSError('status missing')
 h.sdk.GetArmEnableStatus=broken
 with pytest.raises(OSError):h.arm()
 assert h.estopped and not h.armed and h.sdk.calls[-1][0]=='stop'


@pytest.mark.parametrize('mode',['pin','rl'])
def test_tracking_lag_does_not_fault_and_fixed_target_is_kept(mode):
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.mode=mode;e.control('t',{'type':'claim'});e.hardware_action('t','connect');e.hardware_action('t','arm')
 sdk=e.hardware.sdk
 # The plant continues to report its old angles while accepting new commands.
 sdk.JointCtrl=lambda *q:sdk.calls.append(('joints',q))
 desired=e.measured_q.copy();desired[5]+=.2;e.target=e.pose(desired);target=e.target.copy()
 for _ in range(180):
  before=e.q.copy();v=e.limiter.v.copy();e.last_heartbeat=time.monotonic();e.tick()
  assert e.fault is None and e.hardware.armed and not e.hardware.estopped
  assert np.allclose(e.target,target)
  assert np.all(abs(e.q-before)<=e.limiter.vmax*.02+1e-8)
  assert np.max(abs(e.limiter.v-v))<=e.config['max_joint_acceleration_rad_s2']*.02+1e-8
 assert np.max(abs(e.q-e.measured_q))>.06
 count=sum(c[0]=='joints' for c in sdk.calls)
 for _ in range(30):e.last_heartbeat=time.monotonic();e.tick()
 assert sum(c[0]=='joints' for c in sdk.calls)==count+30
 # Feedback catches up; the same target converges without clearing or re-arming.
 sdk.JointCtrl=lambda *q:FakeSDK.JointCtrl(sdk,*q)
 for _ in range(80):e.last_heartbeat=time.monotonic();e.tick()
 assert e.fault is None and e.snapshot()['orientation_error_deg']<.5
 assert np.allclose(e.target,target)


def test_keyboard_target_can_lead_measured_pose_without_latching():
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.control('t',{'type':'claim'});e.hardware_action('t','connect');e.hardware_action('t','arm')
 initial=e.target.copy()
 for _ in range(40):e.last_nudge=-1e10;e.control('t',{'type':'nudge','key':'a'})
 assert abs(e.target[1,3]-initial[1,3]-.04)<1e-9
 for _ in range(20):e.last_nudge=-1e10;e.control('t',{'type':'nudge','key':'q'})
 assert e.fault is None and e.hardware.armed


def test_clear_fault_resets_latch_without_motion_and_archives_history():
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.control('t',{'type':'claim'});e.hardware_action('t','connect');e.hardware_action('t','arm')
 safe=e.safe_q.copy();sdk=e.hardware.sdk;sdk.GetArmEnableStatus=lambda:[False]*6;e.tick()
 assert e.fault and e.hardware.estopped
 reason=e.last_fault['reason'];count=len(sdk.calls)
 e.hardware_action('t','clear')
 assert e.fault is None and not e.hardware.estopped and not e.hardware.armed and e.paused
 assert e.joint_goal is None and e.disable_hold_since is None
 assert e.last_fault['reason']==reason and 'cleared_at' in e.last_fault
 assert np.array_equal(e.safe_q,safe)
 assert np.allclose(e.q,e.hardware.feedback()) and np.max(abs(e.limiter.v))==0
 assert sdk.calls[count:]==[('stop',(2,0,0))] # clear only, never enable or joint command
 e.tick();assert e.fault is None and not e.hardware.estopped


def test_clear_rejects_invalid_feedback_and_does_not_hide_fault():
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.control('t',{'type':'claim'});e.hardware_action('t','connect');e.hardware_action('t','arm')
 e.hardware.sdk.GetArmEnableStatus=lambda:[False]*6;e.tick();reason=e.fault;sdk=e.hardware.sdk
 sdk.stale=True;e.hardware.last_feedback_time=time.monotonic()-1;count=len(sdk.calls)
 with pytest.raises(RuntimeError,match='反馈'):e.hardware_action('t','clear')
 assert e.fault==reason and e.hardware.estopped and len(sdk.calls)==count


def test_waiting_for_initial_feedback_is_not_a_latched_fault():
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.control('t',{'type':'claim'});e.hardware_action('t','connect');sdk=e.hardware.sdk;sdk.stale=True
 for _ in range(3):e.tick()
 assert e.fault is None and e.last_fault is None and not e.hardware.estopped and not e.feedback_valid
 assert '等待有效关节反馈' in e.message
 assert [c[0] for c in sdk.calls]==['connect']
 sdk.stale=False;e.tick();assert e.feedback_valid and e.fault is None
 e.hardware_action('t','arm');e.control('t',{'type':'pause'})
 assert e.fault is None and e.hardware.estopped # a deliberate stop, not a fault

@pytest.mark.parametrize('mode',['pin','rl'])
def test_sdk_velocity_limits_replace_fixed_cap_for_both_solvers(mode):
 from piper_ik.engine import Engine
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,FakeSDK);e.mode=mode;e.control('t',{'type':'claim'})
 e.hardware_action('t','connect');e.hardware_action('t','arm')
 assert np.array_equal(e.limiter.vmax,[3,3,3,3,3,2])
 assert e.snapshot()['sdk_velocity_limits_rad_s']==[3,3,3,3,3,2]
 assert e.reference.linear_speed is None and e.reference.angular_speed is None
 q=e.hardware.last_command.copy();q[0]+=.01 # 0.5 rad/s, above the old 0.25 cap.
 e.hardware.send(q,.02)
 assert e.hardware.sdk.calls[-2]==('mode',(1,1,5,0))
 q[5]+=.05 # 2.5 rad/s exceeds J6's SDK limit, even though other axes allow 3.
 with pytest.raises(RuntimeError,match='rate limit'):e.hardware.send(q,.02)

@pytest.mark.parametrize('bad',['stale','wrong_joint','zero','nan'])
def test_invalid_speed_readback_never_enables_or_sends_motion(monkeypatch,bad):
 clock=SimpleNamespace(now=100.)
 monkeypatch.setattr('piper_ik.hardware.time.monotonic',lambda:clock.now)
 monkeypatch.setattr('piper_ik.hardware.time.sleep',lambda dt:setattr(clock,'now',clock.now+dt))
 class BadSpeedSDK(FakeSDK):
  def SearchMotorMaxAngleSpdAccLimit(self,joint,kind):
   if bad=='stale':return
   super().SearchMotorMaxAngleSpdAccLimit(joint,kind)
   motor=self.speed_reply.current_motor_angle_limit_max_vel
   if bad=='wrong_joint':motor.motor_num=0
   else:motor.max_joint_spd=0 if bad=='zero' else float('nan')
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,BadSpeedSDK);h.connect()
 with pytest.raises(RuntimeError,match='SDK 最大速度'):h.arm()
 assert not h.armed and h.velocity_limits is None
 assert h.sdk.calls==[('connect',{'piper_init':False})]


def test_reenable_refreshes_drive_velocity_and_optional_override():
 config=json.loads((ROOT/'config.json').read_text());config['hardware_enabled']=True
 h=PiperHardware(config,True,FakeSDK);h.connect();h.arm();h.halt();h.clear_stop()
 original=h.sdk.SearchMotorMaxAngleSpdAccLimit
 def changed(joint,kind):
  original(joint,kind);h.sdk.speed_reply.current_motor_angle_limit_max_vel.max_joint_spd=500
 h.sdk.SearchMotorMaxAngleSpdAccLimit=changed;h.arm()
 assert np.array_equal(h.command_velocity_limits(),[.5]*6)
 config['max_joint_velocity_rad_s']=.2
 assert np.array_equal(h.command_velocity_limits(),[.2]*6)
