"""Feedback authority and disconnect behavior; never contacts robot hardware."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from piper_ik.engine import Engine
ROOT=Path(__file__).resolve().parents[1]

def test_selected_simulator_feedback_drives_display_and_disconnect_pauses():
 e=Engine(json.loads((ROOT/'config.json').read_text()))
 class FakeSim:
  backend='isaac';headless=True;process=SimpleNamespace(poll=lambda:None)
  def __init__(self):self.frames=0;self.fail=False;self.shift=.00002
  def step(self,q,target,tcp):
   if self.fail:raise ConnectionError('Simulator feedback disconnected')
   self.frames+=1;T=e.flange(q);T[0,3]+=self.shift
   return {'q':q.tolist(),'flange':T.tolist(),'links':[],'frames':self.frames}
 sim=FakeSim();e.attach_simulator(sim)
 s=e.snapshot();assert s['simulator']=='isaac' and s['simulator_connected']
 assert abs(s['actual']['position'][0]-e.flange(e.q)[0,3]-.00002)<1e-10
 e.control('test',{'type':'claim'});sim.fail=True;e.tick()
 s=e.snapshot();assert s['paused'] and not s['simulator_connected'];assert 'disconnected' in s['fault'];assert not s['hardware_connected']

def test_headless_mujoco_bridge_changes_real_model_and_exits():
 from piper_ik.simulator import SimulatorBridge
 sim=SimulatorBridge('mujoco',headless=True,timeout=20)
 process=sim.process;path=sim.path
 try:
  q=np.array([0.,1.,-1.2,.2,.55,0.]);T=np.eye(4)
  a=sim.step(q,T,T);q[0]+=.1;b=sim.step(q,T,T)
  assert b['frames']>a['frames'];assert abs(b['q'][0]-.1)<1e-10
  assert np.linalg.norm(np.array(a['flange'])[:3,3]-np.array(b['flange'])[:3,3])>.005
  assert len(b['links'])==7
 finally:sim.close()
 assert process.poll() is not None and not path.exists()
