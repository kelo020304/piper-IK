#!/usr/bin/env python3
"""Load standalone USD into Isaac Sim; compare FK and exercise physical mimic.
Run with Isaac Sim's Python. Does not modify the USD.
"""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args,_=p.parse_known_args();root=args.root.resolve()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','width':640,'height':480,'anti_aliasing':0,'multi_gpu':False,'limit_cpu_threads':4,'hide_ui':True})
import numpy as np
import omni.usd
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from scipy.spatial.transform import Rotation as R
world=World(stage_units_in_meters=1.,physics_dt=1/240,rendering_dt=1/30,backend='numpy',device='cpu')
stage=omni.usd.get_context().get_stage();robot=stage.DefinePrim('/World/PiperUMI','Xform');robot.GetReferences().AddReference(str(root/'usd/piper_umi.usd'))
art=world.scene.add(Articulation(prim_paths_expr='/World/PiperUMI/base_link',name='piper_umi'))
for prim in stage.Traverse():
 if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
world.reset();print('ISAAC_JOINTS',art.dof_names,'BODIES',art.body_names,flush=True)
spec=json.loads((root/'metadata/model_spec.json').read_text());physical=[j for j in spec['joints'] if j['type']!='fixed'];order=[art.get_dof_index(j['name']) for j in physical]
assert len(art.dof_names)==8
# Frozen gravity-free FK probe; unchanged collision and mimic configuration.
art.set_gains(kps=np.zeros((1,8)),kds=np.zeros((1,8)))
def forward(q):
 Ts={'base_link':np.eye(4)};qi=dict(zip([j['name'] for j in physical],q))
 for j in spec['joints']:
  t=np.array(j['T']);motion=np.eye(4)
  if j['type']=='revolute':motion[:3,:3]=R.from_rotvec(np.array(j['axis'])*qi[j['name']]).as_matrix()
  elif j['type']=='prismatic':motion[:3,3]=np.array(j['axis'])*qi[j['name']]
  Ts[j['child']]=Ts[j['parent']]@t@motion
 return Ts
# Refresh kinematics before integrating, then compare after one tiny step.
errors=[]
samples=[spec['cad_qpos'],[.2,.5,-.8,.2,.3,-.4,.02,.02],[-.3,.8,-1.2,-.3,-.2,.5,.04,.04]]
for q in samples:
 positions=np.zeros((1,8),dtype=np.float32);positions[0,order]=q
 art.set_joint_positions(positions);art.set_joint_velocities(np.zeros_like(positions));world.step(render=False)
 actual=art.get_joint_positions()[0,order];expected=forward(actual);trans=np.asarray(art._physics_view.get_link_transforms())[0]
 for name in ['link6','left_finger','right_finger']:
  v=trans[art.get_body_index(name)];T=expected[name]
  errors.append({'link':name,'position_m':float(np.linalg.norm(v[:3]-T[:3,3])),'orientation_rad':float(R.from_matrix(R.from_quat(v[3:7]).as_matrix().T@T[:3,:3]).magnitude())})
 print('ISAAC_FK',errors[-3:],flush=True)
# Return to CAD configuration and exercise the actual mimic constraint with gravity and collisions.
world.stop()
for prim in stage.Traverse():
 if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Set(False)
world.reset()
positions=np.zeros((1,8),dtype=np.float32);positions[0,order]=spec['cad_qpos'];art.set_joint_positions(positions);art.set_joint_velocities(np.zeros_like(positions))
kp=np.zeros((1,8));kd=np.zeros((1,8));kp[0,order[:6]]=1200;kd[0,order[:6]]=70;kp[0,order[6]]=1500;kd[0,order[6]]=30
art.set_gains(kps=kp,kds=kd)
sweeps=[]
for travel in [0.,.025,.05,0.]:
 targets=positions.copy();targets[0,order[6]]=travel;art.set_joint_position_targets(targets)
 for _ in range(480):world.step(render=False)
 q=art.get_joint_positions()[0,order];err=abs(float(q[6]-q[7]));sweeps.append({'command_m':travel,'left_m':float(q[6]),'right_m':float(q[7]),'mimic_error_m':err});print('ISAAC_SWEEP',sweeps[-1],flush=True)
report={'isaac_sim_version':__import__('importlib.metadata',fromlist=['version']).version('isaacsim'),'dof_names':art.dof_names,'body_names':art.body_names,'fk_errors':errors,'gripper_sweeps':sweeps,'fk_gravity_disabled':True,'sweep_gravity_disabled':False,'collisions_enabled':True,'max_fk_position_m':max(e['position_m'] for e in errors),'max_fk_orientation_rad':max(e['orientation_rad'] for e in errors),'max_mimic_error_m':max(s['mimic_error_m'] for s in sweeps)}
(root/'validation/isaac_sim.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['max_fk_position_m']<1e-5,report
assert report['max_fk_orientation_rad']<1e-4,report
assert report['max_mimic_error_m']<1e-4,report
assert max(abs(s['left_m']-s['command_m']) for s in sweeps)<1e-3
print('ISAAC_VALIDATION_PASS',flush=True);app.close()
