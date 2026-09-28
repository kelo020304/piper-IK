import json,time
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from piper_ik.engine import Engine,KEYS
from piper_ik.hardware import PiperHardware,HardwareLocked
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def config():return json.loads((ROOT/'config.json').read_text())
@pytest.fixture(scope='module')
def engine():return Engine(json.loads((ROOT/'config.json').read_text()))
def reset(e,mode='pin'):
 e.owner=None;e.control('test',{'type':'claim'});e.control('test',{'type':'reset'});e.control('test',{'type':'claim'});e.control('test',{'type':'mode','mode':mode});e.last_nudge=-1e10
@pytest.mark.parametrize('mode',['pin','rl'])
@pytest.mark.parametrize('key',list(KEYS))
def test_every_keyboard_axis_is_small_and_tracks(engine,mode,key):
 reset(engine,mode);before=engine.target.copy();q0=engine.q.copy();engine.control('test',{'type':'nudge','key':key});target=engine.target.copy();kind,axis,sign=KEYS[key]
 if kind=='p':
  expected=np.zeros(3);expected[axis]=sign*.001;assert np.allclose(target[:3,3]-before[:3,3],expected)
 else:
  rv=Rotation.from_matrix(target[:3,:3]@before[:3,:3].T).as_rotvec();expected=np.zeros(3);expected[axis]=sign*np.deg2rad(.5);assert np.allclose(rv,expected)
 qs=[q0];vel=[np.zeros(6)]
 for _ in range(70):
  engine.last_heartbeat=time.monotonic();engine.tick();qs.append(engine.q.copy());vel.append(engine.limiter.v.copy())
 state=engine.snapshot();assert state['fault'] is None
 assert state['position_error_mm']<.7,(mode,key,state['position_error_mm'])
 assert state['orientation_error_deg']<.4,(mode,key,state['orientation_error_deg'])
 assert np.all(abs(np.diff(qs,axis=0))<=engine.limiter.vmax*.02+1e-8)
 assert np.max(abs(np.diff(vel,axis=0)/.02))<=1.5+1e-6
 assert np.linalg.norm(engine.q-q0)>1e-5

def test_switch_pause_disconnect_and_joint_continuity(engine):
 reset(engine);q0=engine.q.copy();v0=engine.limiter.v.copy()
 for i in range(90):
  engine.last_heartbeat=time.monotonic()
  if i%10==0 and i<50:
   engine.last_nudge=-1e10;engine.control('test',{'type':'nudge','key':'q'})
  if i==30:engine.control('test',{'type':'mode','mode':'rl'})
  if i==50:engine.control('test',{'type':'pause'})
  engine.tick();assert np.all(abs(engine.q-q0)<=engine.limiter.vmax*.02+1e-8);assert max(abs(engine.limiter.v-v0))<=.03+1e-8;q0=engine.q.copy();v0=engine.limiter.v.copy()
 assert max(abs(engine.limiter.v))<1e-8
 engine.control('test',{'type':'claim'});engine.last_heartbeat=time.monotonic()-1;engine.tick();assert engine.paused and engine.owner is None

def test_limit_input_flood_and_lease(engine):
 reset(engine);T=engine.target.copy()
 for _ in range(100):engine.control('test',{'type':'nudge','key':'w'})
 assert np.linalg.norm(engine.target[:3,3]-T[:3,3])<=.001000001
 with pytest.raises(PermissionError):engine.control('other',{'type':'nudge','key':'w'})
 with pytest.raises(ValueError):engine.control('test',{'type':'steps','mm':float('nan'),'deg':.5})
 with pytest.raises(ValueError):engine.control('test',{'type':'steps','mm':20,'deg':.5})

def test_hardware_locked_before_any_sdk_factory(config):
 calls=[]
 def forbidden(*a,**k):calls.append(1);raise AssertionError('SDK constructed')
 h=PiperHardware(config,authorized=True,sdk_factory=forbidden)
 for action in [h.connect,h.arm,h.halt,h.clear_stop,h.disconnect]:
  with pytest.raises(HardwareLocked):action()
 assert calls==[]
 config['hardware_enabled']=True;h=PiperHardware(config,authorized=False,sdk_factory=forbidden)
 with pytest.raises(HardwareLocked):h.connect()
 assert calls==[]

@pytest.mark.parametrize('mode',['pin','rl'])
def test_configured_gripper_tcp_tracks_rotation(config,mode):
 config['tcp_xyz_m']=[.03,.01,.04];e=Engine(config);reset(e,mode);e.control('test',{'type':'nudge','key':'q'})
 for _ in range(100):e.last_heartbeat=time.monotonic();e.tick()
 state=e.snapshot();assert state['position_error_mm']<.7;assert state['orientation_error_deg']<.4;assert not state['fault']
