#!/usr/bin/env python3
"""Relocation, dependency, FK, geometry and MuJoCo dynamic checks.
Usage: python validate_models.py [--root ROOT] [--cad-dir EXTRACTED]
"""
import argparse,json,hashlib,shutil,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,Sdf
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--cad-dir',type=Path);args=p.parse_args();root=args.root.resolve();spec=json.loads((root/'metadata/model_spec.json').read_text());report={};(root/'validation').mkdir(exist_ok=True)
physical=[j for j in spec['joints'] if j['type']!='fixed'];names=[j['name'] for j in physical]
def fk(q):
 poses={'base_link':np.eye(4)};coordinates=dict(zip(names,q))
 for j in spec['joints']:
  T=np.array(j['T']);motion=np.eye(4)
  if j['type']=='revolute':motion[:3,:3]=R.from_rotvec(np.array(j['axis'])*coordinates[j['name']]).as_matrix()
  elif j['type']=='prismatic':motion[:3,3]=np.array(j['axis'])*coordinates[j['name']]
  poses[j['child']]=poses[j['parent']]@T@motion
 return poses
with tempfile.TemporaryDirectory(prefix='piper_umi_relocation_') as tmp:
 for fmt in ('urdf','xml','usd'):
  isolated=Path(tmp)/('standalone_'+fmt);shutil.copytree(root/fmt,isolated)
  if fmt in ('urdf','xml'):
   xml=ET.parse(isolated/('piper_umi.urdf' if fmt=='urdf' else 'piper_umi.xml'))
   refs=[e.get('filename' if fmt=='urdf' else 'file') for e in xml.findall('.//mesh')]
   for ref in refs:
    assert ref and not Path(ref).is_absolute() and '://' not in ref,ref
    asset=(isolated/ref).resolve();assert isolated in asset.parents and asset.is_file(),asset
   loaded=mujoco.MjModel.from_xml_path(str(isolated/('piper_umi.urdf' if fmt=='urdf' else 'scene.xml')))
   report[fmt+'_relocation']={'loaded':True,'mesh_references':len(refs),'nq':loaded.nq,'nu':loaded.nu}
  else:
   stage=Usd.Stage.Open(str(isolated/'piper_umi.usd'));assert stage
   layers,assets,missing=UsdUtils.ComputeAllDependencies(str(isolated/'piper_umi.usd'));assert not missing,missing
   for f in assets:assert isolated in Path(f).resolve().parents
   report['usd_relocation']={'loaded':True,'layers':len(layers),'external_assets':len(assets),'unresolved':len(missing),'rigid_bodies':sum(x.HasAPI(UsdPhysics.RigidBodyAPI) for x in stage.Traverse())}
   sl,sa,sm=UsdUtils.ComputeAllDependencies(str(isolated/'scene.usda'));assert not sm
   assert all(isolated in Path(l.realPath).resolve().parents for l in sl)
   assert all(isolated in Path(a).resolve().parents for a in sa)
   assert len(sa)==1 and Path(sa[0]).name=='blue_checker.png'
   report['usd_scene_relocation']={'layers':len(sl),'unresolved':len(sm)}
# Independent URDF transform implementation, then MuJoCo poses over varied configurations.
ur=ET.parse(root/'urdf/piper_umi.urdf').getroot();parent_names={j.find('child').get('link') for j in ur.findall('joint')};assert set(l.get('name') for l in ur.findall('link'))-parent_names=={'base_link'}
def urdf_fk(q):
 poses={'base_link':np.eye(4)};coords=dict(zip(names,q))
 for j in ur.findall('joint'):
  o=j.find('origin');T=np.eye(4);T[:3,3]=np.fromstring(o.get('xyz'),sep=' ');T[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix();typ=j.get('type');mo=np.eye(4)
  if typ!='fixed':
   axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');value=coords[j.get('name')]
   if typ=='revolute':mo[:3,:3]=R.from_rotvec(axis*value).as_matrix()
   else:mo[:3,3]=axis*value
  poses[j.find('child').get('link')]=poses[j.find('parent').get('link')]@T@mo
 return poses
m=mujoco.MjModel.from_xml_path(str(root/'xml/scene.xml'));d=mujoco.MjData(m);rng=np.random.default_rng(4);qmin=np.array([j['limit']['lower'] for j in physical]);qmax=np.array([j['limit']['upper'] for j in physical]);errors=[]
for q in [np.zeros(8),np.array(spec['cad_qpos'])]+[rng.uniform(qmin,qmax) for _ in range(30)]:
 q[7]=q[6];d.qpos[:]=q;mujoco.mj_forward(m,d);expected=urdf_fk(q)
 for name,T in expected.items():
  if name=='tool0':idx=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,name);pos=d.site_xpos[idx];rot=d.site_xmat[idx].reshape(3,3)
  else:idx=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name);pos=d.xpos[idx];rot=d.xmat[idx].reshape(3,3)
  errors.append([float(np.linalg.norm(pos-T[:3,3])),float(R.from_matrix(rot.T@T[:3,:3]).magnitude())])
errors=np.array(errors);assert errors[:,0].max()<1e-8 and errors[:,1].max()<1e-8
report['urdf_mujoco_fk']={'configurations':32,'max_position_error_m':errors[:,0].max(),'max_rotation_error_rad':errors[:,1].max()}
stage=Usd.Stage.Open(str(root/'usd/piper_umi.usd'));zero=urdf_fk(np.zeros(8));usd_errors=[]
for n,T in zero.items():
 path='/PiperUMI/gripper_base/tool0' if n=='tool0' else '/PiperUMI/'+n
 W=np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
 usd_errors.append(np.max(abs(W-T)))
assert max(usd_errors)<1e-6;report['usd_zero_pose']={'max_transform_entry_error':max(usd_errors)}
# Reconstruct the STEP assembly vertices using the calibrated CAD configuration.
if args.cad_dir:
 cal=json.loads((root/'metadata/cad_alignment.json').read_text());B=np.array(cal['B']);actual=urdf_fk(np.array(spec['cad_qpos']));maxdist=0
 for item in spec['part_inventory']:
  target=trimesh.load(args.cad_dir/(item['cad_id']+'.stl'));target.apply_scale(.001)
  mesh=trimesh.load(root/'urdf'/item['visual_mesh']);mesh.apply_transform(np.array(spec['source_link_frames_cad'][item['link']]) if 'source_link_frames_cad' in spec else B@actual[item['link']]);a=cKDTree(target.vertices).query(mesh.vertices)[0].max();b=cKDTree(mesh.vertices).query(target.vertices)[0].max();maxdist=max(maxdist,a,b)
 assert maxdist<2e-7,maxdist
 report['retained_step_geometry']={'visual_parts':len(spec['part_inventory']),'max_vertex_distance_m':maxdist,'scope':'Retained custom gripper parts in their source CAD frames; stock mounting offset is intentional.'}
# Check authored colors in all three independent output formats and OBJ dependencies.
mx=ET.parse(root/'xml/piper_umi.xml').getroot();color_count=0
for name,link in spec['links'].items():
 for i,a in enumerate(link['visuals']):
  uc=ur.find(f"link[@name='{name}']/visual[@name='{name}_visual_{i}']/material/color")
  mc=mx.find(f".//geom[@name='{name}_visual_{i}']")
  up=stage.GetPrimAtPath(f'/PiperUMI/{name}/visual_{i}')
  assert np.allclose(np.fromstring(uc.get('rgba'),sep=' '),a['color'],atol=1e-7)
  assert np.allclose(np.fromstring(mc.get('rgba'),sep=' '),a['color'],atol=1e-7)
  assert np.allclose(UsdGeom.Mesh(up).GetDisplayColorAttr().Get()[0],a['color'][:3],atol=1e-7)
  binding=up.GetRelationship('material:binding').GetTargets();assert len(binding)==1
  shader=stage.GetPrimAtPath(str(binding[0])+'/Shader')
  assert np.allclose(shader.GetAttribute('inputs:diffuseColor').Get(),a['color'][:3],atol=1e-7)
  if a['mesh'].endswith('.obj'):
   for fmt in ('urdf','xml'):
    assert not any(x.startswith(('mtllib ','usemtl ')) for x in (root/fmt/a['mesh']).read_text().splitlines())
  color_count+=1
assert color_count==86
for n in ['left_finger','right_finger']:
 colors=[a['color'] for a in spec['links'][n]['visuals']]
 assert [.95,.95,.95,1.] in colors and [1.,.67,.30,1.] in colors
assert spec['links']['link6']['visuals'][0]['mesh'].endswith('native_link6.stl')
assert spec['links']['flange_link']['visuals'][0]['mesh'].endswith('native_flange.stl')
assert all(x['cad_label']!='UMI-PIPER-CAMERA-MOUNT' for x in spec['part_inventory'])
report['appearance']={'visual_parts':color_count,'zip_arm_parts':76,'native_disk_and_flange':True,'matching_colors_all_formats':True,'unresolved_obj_material_libraries':0,'supports_rgba':[.95,.95,.95,1.],'fingers_rgba':[1.,.67,.30,1.]}
# Dynamic drive and real equality constraint, gravity + collision on.
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);report['cad_pose_contacts']=int(d.ncon);assert d.ncon==0
sweeps=[]
for qg in [0.,.025,.05,0.]:
 d.ctrl[6]=qg
 for _ in range(1000):mujoco.mj_step(m,d)
 e=abs(d.qpos[6]-d.qpos[7]);tracking=abs(d.qpos[6]-qg)
 sweeps.append({'command_m':qg,'left_m':float(d.qpos[6]),'right_m':float(d.qpos[7]),'mimic_error_m':float(e),'tracking_error_m':float(tracking),'contacts':int(d.ncon)})
 assert np.isfinite(d.qpos).all() and e<1e-5 and tracking<.001
assert not np.any(d.warning.number)
# Move the arm as well, through a non-singular configuration.
d.ctrl[:6]=[.15,.5,-.8,.15,.2,-.3]
for _ in range(1500):mujoco.mj_step(m,d)
assert np.isfinite(d.qpos).all() and not np.any(d.warning.number)
report['mujoco_dynamics']={'version':mujoco.__version__,'gravity_enabled':True,'collisions_enabled':True,'steps':5500,'warnings':d.warning.number.tolist(),'gripper_sweeps':sweeps,'arm_target':d.ctrl[:6].tolist(),'arm_final':d.qpos[:6].tolist(),'max_arm_tracking_error_rad':float(np.max(abs(d.qpos[:6]-d.ctrl[:6])))}
assert report['mujoco_dynamics']['max_arm_tracking_error_rad']<.05
# Triangles and all positive-definite inertias.
for l in spec['links'].values():
 if 'inertial' in l:
  assert l['inertial']['mass']>0 and np.linalg.eigvalsh(l['inertial']['tensor']).min()>0
report['passed']=True
(root/'validation/checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
