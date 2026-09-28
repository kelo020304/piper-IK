from fastapi.testclient import TestClient
from piper_ik.server import create_app

def test_default_app_cannot_open_hardware_and_standalone_docs_load():
 calls=[]
 def fail(*a,**kw):calls.append(1);raise AssertionError('Hardware access')
 app=create_app(sdk_factory=fail)
 with TestClient(app,base_url='http://127.0.0.1:8765') as client:
  assert client.get('/api/state').json()['hardware_allowed'] is False
  assert client.post('/api/hardware/connect').status_code==403
  with client.websocket_connect('ws://127.0.0.1:8765/ws',headers={'origin':'http://127.0.0.1:8765'}) as ws:
   hello=ws.receive_json();assert hello['type']=='hello';ws.send_json({'type':'claim'})
   ws.send_json({'type':'hardware','action':'connect'})
   for _ in range(10):
    r=ws.receive_json()
    if r['type']=='error':assert '锁定' in r['message'];break
   else:raise AssertionError('Hardware request was not rejected')
  for path in ['/guide/','/report/','/hardware-guide/','/sim-guide/','/docs/RL.html','/docs/PINOCCHIO.html']:
   r=client.get(path)
   assert r.status_code==200 and 'text/html' in r.headers['content-type']
 assert calls==[]


def test_runtime_hardware_flags_do_not_write_config_or_connect(tmp_path):
 import json
 from pathlib import Path
 original=(Path(__file__).resolve().parents[1]/'config.json').read_text()
 path=tmp_path/'config.json';path.write_text(original)
 calls=[]
 def forbidden(*a,**kw):calls.append(1);raise AssertionError('Must not construct SDK at startup')
 app=create_app(path,authorized=True,can_interface='can_test',sdk_factory=forbidden)
 assert app.state.engine.hardware.authorized
 assert app.state.engine.config['can_interface']=='can_test'
 assert app.state.engine.hardware.sdk is None and not calls
 assert path.read_text()==original
 # Even a saved legacy true flag must not authorize the next process by itself.
 config=json.loads(original);config['hardware_enabled']=True;path.write_text(json.dumps(config))
 app=create_app(path,can_interface='can0',sdk_factory=forbidden)
 assert not app.state.engine.hardware.authorized and not calls
 assert json.loads(path.read_text())['hardware_enabled'] is True


def test_live_transport_keeps_control_without_browser_timer_and_close_stops():
 import time
 import numpy as np
 from test_hardware import FakeSDK
 app=create_app(authorized=True,sdk_factory=FakeSDK)
 with TestClient(app,base_url='http://127.0.0.1:8765') as client:
  with client.websocket_connect('ws://127.0.0.1:8765/ws?source=web',headers={'origin':'http://127.0.0.1:8765'}) as ws:
   ws.receive_json()
   for msg in ({'type':'claim'},{'type':'hardware','action':'connect'},{'type':'hardware','action':'arm'},{'type':'nudge','key':'w'}):
    ws.send_json(msg)
    while True:
     response=ws.receive_json();assert response['type']!='error',response
     if response['type']=='ack':break
   engine=app.state.engine;goal=engine.target.copy();sdk=engine.hardware.sdk
   # No JS/application heartbeat: transport is still open in a background tab.
   time.sleep(1.1)
   assert engine.hardware.armed and not engine.paused and engine.fault is None
   assert np.allclose(engine.target,goal)
  for _ in range(100):
   if not engine.hardware.armed:break
   time.sleep(.01)
  assert not engine.hardware.armed and engine.hardware.estopped and engine.owner is None
  assert any(c[0]=='stop' for c in sdk.calls)


def test_enable_failure_remains_in_state_after_websocket_error():
 from test_hardware import FakeSDK
 class RefusedSDK(FakeSDK):
  def EnableArm(self,n):raise OSError('使能发送失败（测试）')
 app=create_app(authorized=True,sdk_factory=RefusedSDK)
 with TestClient(app,base_url='http://127.0.0.1:8765') as client:
  with client.websocket_connect('ws://127.0.0.1:8765/ws',headers={'origin':'http://127.0.0.1:8765'}) as ws:
   ws.receive_json()
   for action in ('claim','connect','clear','arm'):
    ws.send_json({'type':'claim'} if action=='claim' else {'type':'hardware','action':action})
    while True:
     msg=ws.receive_json()
     if msg['type'] in ('error','ack'):break
    assert msg['type']==('error' if action=='arm' else 'ack')
   state=client.get('/api/state').json()
   assert state['hardware_estopped'] and not state['hardware_armed']
   assert '使能发送失败' in state['fault'] and state['message']==state['fault']
   assert state['last_fault']['action']=='arm' and state['safe_q'] is None
