import json,time
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from piper_ik.trajectory import JointTrajectory,PoseReference
from piper_ik.engine import Engine

def test_fixed_joint_step_brakes_before_target_without_overshoot():
    t=JointTrajectory([-1]*6,[1]*6,np.zeros(6),.25,1.,10.)
    rows=[]
    for _ in range(150):rows.append(t.apply(np.full(6,.005),.02))
    rows=np.array(rows)
    assert np.min(np.diff(rows,axis=0))>=-1e-12
    assert np.max(rows)<=.005+1e-12 and np.allclose(rows[-1],.005)

def test_retargets_and_irregular_dt_preserve_velocity_acceleration_jerk_limits():
    t=JointTrajectory([-1]*6,[1]*6,np.zeros(6),.25,1.,10.)
    for k in range(240):
        dt=[.018,.021,.04,.022][k%4]
        goal=np.full(6,.15 if k<35 else -.12 if k<100 else .08)
        v,a=t.velocity.copy(),t.a.copy();q=t.q.copy()
        t.apply(goal,dt)
        assert np.max(abs(t.velocity))<=.25+1e-9
        assert np.max(abs(t.q-q))<=.25*dt+1e-9
        assert np.max(abs(t.a))<=1+1e-9
        assert np.max(abs(t.velocity-v))<=dt+1e-9
        assert np.max(abs(t.a-a))<=10*dt+1e-8
        assert np.max(abs(t.q))<1
    assert np.allclose(t.q,.08,atol=1e-7)

def test_reference_rotation_uses_short_arc_and_limits_translation():
    initial=np.eye(4);initial[:3,:3]=Rotation.from_euler('z',179,degrees=True).as_matrix()
    goal=initial.copy();goal[:3,:3]=Rotation.from_euler('z',-179,degrees=True).as_matrix();goal[0,3]=.005
    t=PoseReference(initial)
    prev=initial;distance=0.
    for _ in range(150):
        pose=t.step(goal,.02)
        dp=np.linalg.norm(pose[:3,3]-prev[:3,3]);dr=Rotation.from_matrix(pose[:3,:3]@prev[:3,:3].T).magnitude()
        assert dp<=.04*.02+1e-12 and dr<=.5*.02+1e-12
        distance+=dr;prev=pose
    assert np.allclose(pose,goal,atol=1e-9)
    assert abs(distance-np.deg2rad(2))<1e-9

@pytest.mark.parametrize('mode',['pin','rl'])
@pytest.mark.parametrize('key',['w','a','q'])
def test_five_mm_or_degree_step_is_continuous_and_settles(mode,key):
    config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
    e=Engine(config);e.mode=mode;e.control('t',{'type':'claim'});e.control('t',{'type':'steps','mm':5,'deg':5})
    e.control('t',{'type':'nudge','key':key});goal=e.target.copy();positions=[]
    for _ in range(250):
        a=e.limiter.a.copy();v=e.limiter.velocity.copy()
        e.last_heartbeat=time.monotonic();e.tick(.02)
        assert e.fault is None,e.fault
        assert np.max(abs(e.limiter.a-a))<=.2+1e-8
        assert np.max(abs(e.limiter.velocity-v))<=.02+1e-8
        positions.append(e.pose(e.q)[:3,3])
    state=e.snapshot()
    assert state['position_error_mm']<.3 and state['orientation_error_deg']<.3
    assert np.allclose(e.target,goal)
    if key!='q':
        axis=0 if key=='w' else 1
        assert max(p[axis]-goal[axis,3] for p in positions)<.0001

def test_pose_target_validated_and_uses_same_reference_pipeline():
    config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
    e=Engine(config);e.control('t',{'type':'claim'});q=e.q.copy();T=e.target.copy();T[0,3]+=.005
    msg={'type':'pose_target','frame':'base','position_m':T[:3,3].tolist(),'quaternion_xyzw':Rotation.from_matrix(T[:3,:3]).as_quat().tolist()}
    with pytest.raises(PermissionError):e.control('other',msg)
    for bad in ({'frame':'camera'},{'quaternion_xyzw':[0,0,0,0]},{'position_m':[float('nan'),0,0]}):
        with pytest.raises(ValueError):e.control('t',{**msg,**bad})
    e.control('t',msg);assert np.array_equal(e.q,q)
    e.last_heartbeat=time.monotonic();e.tick()
    assert e.reference_pose[0,3]<e.target[0,3]
    e.control('t',{'type':'pause'})
    with pytest.raises(RuntimeError):e.control('t',msg)


def test_per_axis_drive_limits_allow_speed_above_old_cap():
    limits=np.array([.3,.4,.5,.6,.7,.8])
    t=JointTrajectory([-3]*6,[3]*6,np.zeros(6),limits,1.,10.)
    peak=np.zeros(6)
    for _ in range(350):
        t.apply(np.full(6,2.),.02)
        assert np.all(abs(t.velocity)<=limits+1e-8)
        peak=np.maximum(peak,abs(t.velocity))
    assert max(peak)>.25 and np.allclose(t.q,2.,atol=.05)


def test_uncapped_pose_reference_still_filters_and_uses_short_arc():
    pose=np.eye(4);goal=pose.copy();goal[0,3]=.1
    goal[:3,:3]=Rotation.from_euler('z',-179,degrees=True).as_matrix()
    t=PoseReference(pose,linear_speed=None,angular_speed=None)
    prev=pose;max_v=0.;max_w=0.;distance=0.
    for _ in range(100):
        new=t.step(goal,.02)
        max_v=max(max_v,np.linalg.norm(new[:3,3]-prev[:3,3])/.02)
        dr=Rotation.from_matrix(new[:3,:3]@prev[:3,:3].T).magnitude()
        max_w=max(max_w,dr/.02);distance+=dr;prev=new
    assert max_v>.04 and max_w>.5
    assert np.allclose(new,goal,atol=1e-9) and abs(distance-np.deg2rad(179))<1e-9
