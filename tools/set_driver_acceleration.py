"""Explicit, one-shot SDK acceleration setting while the console holds still.
No enable, stop, zero, gripper or joint-motion commands. Defaults to read-only.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def read_acc(sdk,joint):
    previous=sdk.GetCurrentMotorMaxAccLimit().time_stamp
    sdk.SearchMotorMaxAngleSpdAccLimit(joint,0x02)
    deadline=time.monotonic()+1.5
    while time.monotonic()<deadline:
        msg=sdk.GetCurrentMotorMaxAccLimit();value=msg.current_motor_max_acc_limit
        if msg.time_stamp>previous and value.joint_motor_num==joint:
            return int(value.max_joint_acc)
        time.sleep(.01)
    raise RuntimeError(f'J{joint} acceleration query timed out')


def write_acc(sdk,value,guard,audit,save):
    # Readback is recorded raw: installed SDK's feedback-unit comment differs
    # from its setter-unit comment. Never silently apply a x10 conversion.
    audit['before_raw']=[read_acc(sdk,i) for i in range(1,7)];save()
    if any(old<value for old in audit['before_raw']):
        raise RuntimeError('Current raw limit is lower than requested; refusing to increase it')
    for i,old in enumerate(audit['before_raw'],1):
        guard()
        if old!=value:
            sdk.JointMaxAccConfig(i,value)
            audit['writes'].append({'joint':i,'argument':value});save()
            # Official example requires settling time for the drive Flash write.
            for _ in range(10):time.sleep(.05);guard()
        actual=read_acc(sdk,i)
        audit['readback_raw'].append(actual);save()
        if actual!=value:raise RuntimeError(f'J{i}: requested {value}, readback {actual}; stopped before remaining writes')
    audit['verified_raw']=[read_acc(sdk,i) for i in range(1,7)]
    if audit['verified_raw']!=[value]*6:raise RuntimeError('Final readback mismatch')
    audit['status']='verified';save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--can',default='can0');parser.add_argument('--value',type=int,default=100)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--audit',default=str(Path(__file__).resolve().parents[1]/'artifacts/driver-acceleration-sdk.json'))
    args=parser.parse_args()
    if not 1<=args.value<=500:parser.error('--value must be 1–500 (setter units 0.01 rad/s²)')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));baseline=None
    def guard():
        nonlocal baseline
        with opener.open(f'http://127.0.0.1:{args.port}/api/state',timeout=1) as response:s=json.load(response)
        if s['execution']!='hardware' or not s['hardware_connected'] or s['fault'] or s['motion_busy'] or not s['feedback_valid'] or s['feedback_age_ms']>250:
            raise RuntimeError('Console is not in a valid stationary hardware state')
        if max(map(abs,s['velocity']))>.001 or max(abs(a-b) for a,b in zip(s['q'],s['command_q']))>.0035 or s['position_error_mm']>.3 or s['orientation_error_deg']>.3:
            raise RuntimeError('Motion/target change detected; no further parameter writes')
        if baseline is None:baseline=s['q']
        if max(abs(a-b) for a,b in zip(s['q'],baseline))>.001:
            raise RuntimeError('Measured joint motion detected; no further parameter writes')
    if args.apply:
        for _ in range(10):guard();time.sleep(.05)
    # User authorizes only acceleration configuration, not enabling or motion.
    path=Path(args.audit);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError(f'Audit already exists: {path}; choose a new --audit file')
    from piper_sdk import C_PiperInterface_V2
    sdk=C_PiperInterface_V2(args.can)
    audit={'time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'can':args.can,'requested_setter_raw':args.value,
           'setter_rad_s2':args.value*.01,'status':'started','writes':[],'readback_raw':[],
           'scope':'SDK acceleration queries/configuration only; no enable/stop/joint motion sent',
           'units_note':'Setter documented as 0.01 rad/s²; feedback comment says 0.001. Raw readback is preserved without conversion.'}
    def save():path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    try:
        sdk.ConnectPort(piper_init=False)
        if args.apply:write_acc(sdk,args.value,guard,audit,save)
        else:audit['readback_raw']=[read_acc(sdk,i) for i in range(1,7)];audit['status']='read_only';save()
    except Exception as exc:
        audit['status']='failed';audit['error']=str(exc);save();raise
    finally:sdk.DisconnectPort()
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
