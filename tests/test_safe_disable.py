"""No CAN access: safe-return timing, cancellation and controller handover."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from piper_ik.engine import Engine
from test_hardware import FakeSDK

class SafeSDK(FakeSDK):
 def __init__(self,*a,**kw):super().__init__();self.enabled=False;self.disabled_at=[]
 def EnableArm(self,n):super().EnableArm(n);self.enabled=True
 def GetArmEnableStatus(self):return [self.enabled]*6
 def DisableArm(self,n):
  import time
  self.calls.append(('disable',n));self.disabled_at.append(time.monotonic());self.enabled=False

@pytest.fixture
def rig(monkeypatch):
 clock=SimpleNamespace(now=100.)
 monkeypatch.setattr('piper_ik.engine.time.monotonic',lambda:clock.now)
 config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text());config['hardware_enabled']=True
 e=Engine(config,True,SafeSDK)
 e.control('terminal',{'type':'claim','source':'terminal'});e.hardware_action('terminal','connect');e.hardware_action('terminal','arm')
 sdk=e.hardware.sdk
 def tick(n=1):
  for _ in range(n):
   clock.now+=.02;e.last_heartbeat=clock.now;e.tick()
 return e,sdk,clock,tick

@pytest.mark.parametrize('mode',['pin','rl'])
def test_home_then_return_to_first_enable_pose_hold_three_seconds_disable(rig,mode):
 e,sdk,clock,tick=rig;e.mode=mode;safe=e.safe_q.copy()
 e.hardware_action('terminal','home')
 for _ in range(1200):
  tick();assert e.fault is None
  if e.joint_goal is None:break
 assert e.joint_goal is None and max(abs(e.measured_q))<.0035
 e.hardware_action('terminal','disable');requested_at=clock.now
 for _ in range(1200):
  previous=e.q.copy();v=e.limiter.v.copy();tick();assert e.fault is None
  assert np.max(abs(e.q-previous))<=.25*.02+1e-8
  assert np.max(abs(e.limiter.v-v))<=e.config['max_joint_acceleration_rad_s2']*.02+1e-8
  assert not sdk.disabled_at
  if e.motion_kind=='disable_wait':break
 assert e.motion_kind=='disable_wait';arrived_at=e.disable_hold_since
 assert arrived_at-requested_at>3 # countdown starts after return, not after clicking
 tick(149);assert not sdk.disabled_at and e.hardware.armed
 tick(2);assert len(sdk.disabled_at)==1 and sdk.disabled_at[0]-arrived_at>=3
 assert np.max(abs(e.measured_q-safe))<.0035 and e.motion_kind is None and e.paused
 assert e.hardware.estopped and not e.hardware.armed
 count=len([c for c in sdk.calls if c[0]=='joints']);tick(200)
 assert len([c for c in sdk.calls if c[0]=='joints'])==count
 # An emergency recovery must not redefine the recorded first-enable pose.
 sdk.units[0]+=500;e.hardware_action('terminal','clear');e.hardware_action('terminal','arm')
 assert np.array_equal(e.safe_q,safe)

@pytest.mark.parametrize('when',['return','countdown'])
@pytest.mark.parametrize('interrupt',['stop','release','takeover','stale','lost_enable','timeout','disconnect'])
def test_interrupt_cancels_pending_disable(rig,when,interrupt):
 e,sdk,clock,tick=rig
 if when=='return':e.safe_q[0]+=.02
 e.hardware_action('terminal','disable');tick(3)
 assert (e.motion_kind=='disable_wait')==(when=='countdown')
 if interrupt=='stop':e.hardware_action('terminal','stop')
 elif interrupt=='release':e.control('terminal',{'type':'release'})
 elif interrupt=='takeover':e.control('web',{'type':'takeover','source':'web'})
 elif interrupt=='stale':sdk.stale=True;clock.now+=.3;tick()
 elif interrupt=='lost_enable':sdk.enabled=False;tick()
 elif interrupt=='timeout':clock.now+=1;e.tick()
 else:e.hardware_action('terminal','disconnect')
 assert e.motion_kind is None and e.joint_goal is None and e.disable_hold_since is None
 tick(200);assert not sdk.disabled_at


def test_measured_drift_restarts_countdown_and_conflicting_actions_are_blocked(rig):
 e,sdk,clock,tick=rig;e.hardware_action('terminal','disable');tick(5);first=e.disable_hold_since
 for msg in ({'type':'claim'},{'type':'mode','mode':'rl'},{'type':'nudge','key':'w'}):
  with pytest.raises(RuntimeError):e.control('terminal',msg)
 with pytest.raises(RuntimeError):e.hardware_action('terminal','home')
 with pytest.raises(RuntimeError):e.hardware_action('terminal','disable')
 tick(120);sdk.units[0]+=300;tick();assert e.disable_hold_since is None
 tick(5);second=e.disable_hold_since;assert second>first
 tick(35);assert clock.now-first>3 and not sdk.disabled_at
 tick(120);assert len(sdk.disabled_at)==1 and sdk.disabled_at[0]-second>=3


def test_command_arrival_without_measured_arrival_never_disables(rig):
 e,sdk,clock,tick=rig;e.safe_q[0]+=.02
 sdk.JointCtrl=lambda *args:sdk.calls.append(('joints',args)) # stalled arm
 e.hardware_action('terminal','disable');tick(500)
 assert np.max(abs(e.q-e.safe_q))<.0035
 assert np.max(abs(e.measured_q-e.safe_q))>.0035
 assert e.disable_hold_since is None and not sdk.disabled_at
 clock.now+=91;e.last_heartbeat=clock.now;e.tick()
 assert e.fault and e.motion_kind is None and not sdk.disabled_at


def test_takeover_revokes_old_owner_and_preserves_safe_pose(rig):
 e,sdk,clock,tick=rig;safe=e.safe_q.copy()
 with pytest.raises(PermissionError):e.control('web',{'type':'claim'})
 e.control('web',{'type':'takeover','source':'web'})
 assert e.owner=='web' and e.owner_kind=='web' and e.paused and e.hardware.estopped
 assert np.array_equal(e.safe_q,safe)
 for msg in ({'type':'heartbeat'},{'type':'nudge','key':'w'}):
  with pytest.raises(PermissionError):e.control('terminal',msg)
 with pytest.raises(PermissionError):e.hardware_action('terminal','arm')
 e.hardware_action('web','clear');e.hardware_action('web','arm');assert e.hardware.armed
 assert np.array_equal(e.safe_q,safe)
 e.hardware_action('web','disconnect');assert e.safe_q is None


def test_no_auto_enable_for_disable(rig):
 e,sdk,clock,tick=rig;e.hardware_action('terminal','stop');count=len(sdk.calls)
 with pytest.raises(RuntimeError):e.hardware_action('terminal','disable')
 assert len(sdk.calls)==count and e.joint_goal is None


class PersistentStopSDK(SafeSDK):
 """Controller stop survives a client restart; drive enable does not clear it."""
 def __init__(self):super().__init__();self.controller_stopped=False
 def MotionCtrl_1(self,*args):
  super().MotionCtrl_1(*args)
  if args[0]==1:self.controller_stopped=True
  elif args[0]==2:self.controller_stopped=False
 def JointCtrl(self,*args):
  if self.controller_stopped or not self.enabled:self.calls.append(('ignored_joints',args))
  else:super().JointCtrl(*args)


@pytest.mark.parametrize('mode',['pin','rl'])
def test_safe_disable_restart_enable_and_home_resumes_controller(monkeypatch,mode):
 clock=SimpleNamespace(now=100.)
 monkeypatch.setattr('piper_ik.engine.time.monotonic',lambda:clock.now)
 config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text());config['hardware_enabled']=True
 sdk=PersistentStopSDK()
 def new_session():
  e=Engine(config,True,lambda *a,**kw:sdk);e.mode=mode
  e.control('web',{'type':'claim'});e.hardware_action('web','connect')
  return e
 def tick(e):
  clock.now+=.02;e.last_heartbeat=clock.now;e.tick();assert e.fault is None
 first=new_session();first.hardware_action('web','arm');safe=first.safe_q.copy()
 first.hardware_action('web','disable')
 for _ in range(200):
  tick(first)
  if not first.hardware.armed:break
 assert not sdk.enabled and sdk.controller_stopped
 assert np.allclose(first.measured_q,safe)
 first.hardware_action('web','disconnect')
 count=len(sdk.calls);second=new_session()
 assert sdk.calls[count:]==[('connect',{'piper_init':False})]
 assert sdk.controller_stopped and not second.hardware.estopped
 second.hardware_action('web','arm')
 assert sdk.enabled and not sdk.controller_stopped
 assert sdk.calls[count+1:]==[('stop',(2,0,0)),('enable',7)]
 assert np.allclose(second.q,safe) # enabling holds actual pose, no home until requested
 # Home is independent of both IK solvers, even in RL mode.
 def unexpected_solver(*a,**kw):raise AssertionError('home called IK')
 second.pin.solve=unexpected_solver;second.rl.solve=unexpected_solver
 second.hardware_action('web','home')
 for _ in range(1200):
  tick(second)
  if second.joint_goal is None:break
 assert second.joint_goal is None and max(abs(second.measured_q))<.0035
 assert not any(c[0]=='ignored_joints' for c in sdk.calls)


def test_explicit_clear_resets_controller_even_when_local_latch_is_false():
 config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text());config['hardware_enabled']=True
 sdk=PersistentStopSDK();sdk.controller_stopped=True
 e=Engine(config,True,lambda *a,**kw:sdk)
 e.control('web',{'type':'claim'});e.hardware_action('web','connect')
 assert not e.hardware.estopped
 count=len(sdk.calls);e.hardware_action('web','clear')
 assert sdk.calls[count:]==[('stop',(2,0,0))]
 assert not sdk.controller_stopped and not sdk.enabled and not e.hardware.armed
 assert e.paused and e.joint_goal is None
