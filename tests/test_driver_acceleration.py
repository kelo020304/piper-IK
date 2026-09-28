import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('driver_acc',Path(__file__).resolve().parents[1]/'tools/set_driver_acceleration.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ParameterSDK:
 def __init__(self):
  self.values=[500]*6;self.writes=[]
  self.msg=SimpleNamespace(time_stamp=0,current_motor_max_acc_limit=SimpleNamespace(joint_motor_num=0,max_joint_acc=0))
 def SearchMotorMaxAngleSpdAccLimit(self,joint,kind):
  assert kind==2;self.msg.time_stamp+=1
  self.msg.current_motor_max_acc_limit=SimpleNamespace(joint_motor_num=joint,max_joint_acc=self.values[joint-1])
 def GetCurrentMotorMaxAccLimit(self):return self.msg
 def JointMaxAccConfig(self,joint,value):self.writes.append((joint,value));self.values[joint-1]=value

def test_sdk_parameter_write_verified_without_any_motion_api(monkeypatch):
 sleeps=[];monkeypatch.setattr(module.time,'sleep',sleeps.append)
 sdk=ParameterSDK();audit={'writes':[],'readback_raw':[]}
 module.write_acc(sdk,100,lambda:None,audit,lambda:None)
 assert sdk.writes==[(i,100) for i in range(1,7)]
 assert audit['before_raw']==[500]*6 and audit['verified_raw']==[100]*6
 assert audit['status']=='verified' and sum(sleeps)==pytest.approx(3.)

def test_movement_aborts_before_next_write(monkeypatch):
 monkeypatch.setattr(module.time,'sleep',lambda dt:None)
 sdk=ParameterSDK();audit={'writes':[],'readback_raw':[]}
 def guard():
  if sdk.writes:raise RuntimeError('moving')
 with pytest.raises(RuntimeError,match='moving'):module.write_acc(sdk,100,guard,audit,lambda:None)
 assert sdk.writes==[(1,100)]

def test_mismatch_is_not_accepted_or_retried(monkeypatch):
 monkeypatch.setattr(module.time,'sleep',lambda dt:None)
 sdk=ParameterSDK();sdk.JointMaxAccConfig=lambda j,v:sdk.writes.append((j,v))
 with pytest.raises(RuntimeError,match='readback 500'):
  module.write_acc(sdk,100,lambda:None,{'writes':[],'readback_raw':[]},lambda:None)
 assert sdk.writes==[(1,100)]
