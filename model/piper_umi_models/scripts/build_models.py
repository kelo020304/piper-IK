#!/usr/bin/env python3
"""Generate independent URDF, MuJoCo MJCF and USD packages from extracted STEP.
Usage: python build_models.py --cad-dir EXTRACTED --official-dir AGX_ARM_URDF --out OUTPUT
See README.html for provenance, assumptions, and rebuild commands.
"""
import argparse,copy,hashlib,json,shutil,math,re
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot
import xml.etree.ElementTree as ET
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf,Gf,Vt
P=argparse.ArgumentParser();P.add_argument('--cad-dir',type=Path,required=True);P.add_argument('--official-dir',type=Path,required=True);P.add_argument('--out',type=Path,required=True);P.add_argument('--textured-dir',type=Path,required=True);args=P.parse_args()
OUT=args.out.resolve();CAD=args.cad_dir.resolve();OFF=args.official_dir.resolve();TEXTURED=args.textured_dir.resolve()
previous_meshes=set()
if (OUT/'metadata/model_spec.json').exists():
 previous=json.loads((OUT/'metadata/model_spec.json').read_text())
 previous_meshes={a['mesh'] for l in previous['links'].values() for k in ['visuals','collisions'] for a in l[k]}
for fmt in ('urdf','xml','usd'):(OUT/fmt).mkdir(parents=True,exist_ok=True)
for fmt in ('urdf','xml'):(OUT/fmt/'meshes/visual').mkdir(parents=True,exist_ok=True);(OUT/fmt/'meshes/collision').mkdir(parents=True,exist_ok=True)
raw=json.loads((CAD/'parts.json').read_text());parts=raw['parts'];byid={x['name']:x for x in parts}
cal=json.loads((OUT/'metadata/cad_alignment.json').read_text());Tc=[np.array(t) for t in cal['T_cad_from_links']];B=np.array(cal['B']);qcad=np.array(cal['q_cad'])
def arr(s):return np.fromstring(s,sep=' ')
def vals(a):return ' '.join(f'{float(x):.10g}' for x in np.asarray(a).flatten())
def tf(xyz=(0,0,0),rpy=(0,0,0)):
 t=np.eye(4);t[:3,3]=xyz;t[:3,:3]=Rot.from_euler('xyz',rpy).as_matrix();return t
def quat(R):
 q=Rot.from_matrix(R).as_quat();return np.r_[q[3],q[:3]]
def inertia_dict(el):
 o=el.find('origin');v=el.find('inertia');I=np.array([[float(v.get('ixx')),float(v.get('ixy')),float(v.get('ixz'))],[float(v.get('ixy')),float(v.get('iyy')),float(v.get('iyz'))],[float(v.get('ixz')),float(v.get('iyz')),float(v.get('izz'))]])
 return {'mass':float(el.find('mass').get('value')),'com':arr(o.get('xyz')).tolist(),'tensor':I.tolist(),'source':'AgileX official nominal inertial parameters'}
up=ET.parse(OFF/'piper/urdf/piper_description.urdf').getroot()
names=['base_link']+[f'link{i}' for i in range(1,7)]
links={n:{'name':n,'visuals':[],'collisions':[],'inertial':inertia_dict(up.find(f"link[@name='{n}']/inertial"))} for n in names}
joints=[]
for i in range(1,7):
 el=up.find(f"joint[@name='joint{i}']");o=el.find('origin');lim=el.find('limit');joints.append({'name':f'joint{i}','type':'revolute','parent':el.find('parent').get('link'),'child':el.find('child').get('link'),'T':tf(arr(o.get('xyz')),arr(o.get('rpy'))).tolist(),'axis':arr(el.find('axis').get('xyz')).tolist(),'limit':{k:float(lim.get(k)) for k in ['lower','upper','effort','velocity']}})
# CAD-current transforms; geometry is rebased into official local link frames.
Tcad=dict(zip(names,Tc));Tcad['flange_link']=Tc[6].copy()
# The original custom 8 mm flange located the drive 1.487 mm ahead of J6.
# Rebase the retained 155 mm-rail gripper on its drive mounting face, then
# mount it through the stock 10.5 mm flange with the official +4.5 mm joint.
inv6=np.linalg.inv(Tc[6]);drive=trimesh.load(CAD/'Part__Feature008.stl');drive.apply_scale(.001);drive.apply_transform(inv6)
motor=trimesh.load(CAD/'Part__Feature008_solid_0.stl');motor.apply_scale(.001);motor.apply_transform(inv6)
cad_drive_origin=np.r_[motor.bounds.mean(0)[:2],drive.bounds[0,2]]
Tcad['gripper_base']=Tc[6]@tf(cad_drive_origin)
stock=ET.parse(OFF/'piper/urdf/piper_with_gripper_description.xacro').getroot()
links['flange_link']={'name':'flange_link','visuals':[],'collisions':[],'inertial':inertia_dict(stock.find("link[@name='flange_link']/inertial"))}
joints.append({'name':'flange_joint','type':'fixed','parent':'link6','child':'flange_link','T':np.eye(4).tolist()})
links['gripper_base']={'name':'gripper_base','visuals':[],'collisions':[]}
joints.append({'name':'gripper_mount','type':'fixed','parent':'flange_link','child':'gripper_base','T':tf([0,0,.0045]).tolist()})
# Native slider is 22.5 mm long. Keep 0.2 mm separation at model minimum travel.
left=trimesh.load(CAD/'Part__Feature009.stl');right=trimesh.load(CAD/'Part__Feature010.stl')
lc=left.bounds.mean(0)*.001;rc=right.bounds.mean(0)*.001
slider_length=max(left.extents[0],right.extents[0])*.001
q_ref=(rc[0]-lc[0]-slider_length-.0002)/2
assert 0<q_ref<.05
axes_cad={'left_finger':np.array([-1.,0,0]),'right_finger':np.array([1.,0,0])}
for n,c in [('left_finger',lc),('right_finger',rc)]:
 t=Tc[6].copy();t[:3,3]=c;Tcad[n]=t
 closed=t.copy();closed[:3,3]-=axes_cad[n]*q_ref
 jt=np.linalg.inv(Tcad['gripper_base'])@closed
 ax=t[:3,:3].T@axes_cad[n]
 j={'name':'gripper_'+n.split('_')[0]+'_joint','type':'prismatic','parent':'gripper_base','child':n,'T':jt.tolist(),'axis':ax.tolist(),'limit':{'lower':0.,'upper':.05,'effort':10.,'velocity':.2}}
 if n=='right_finger':j['mimic']={'joint':'gripper_left_joint','multiplier':1.,'offset':0.}
 joints.append(j);links[n]={'name':n,'visuals':[],'collisions':[]}
# Exact CAD pieces, including a geometry-conserving partition of the J6 output disk.
mapping={
 'Part__Feature':('base_link','base'), 'Part__Feature001':('link1','shoulder'),
 'Part__Feature002':('link2','upper_arm'), 'Part__Feature003':('link3','forearm'),
 'Part__Feature004':('link4','wrist_roll'), 'Part__Feature005':('link5','wrist_housing'),
 'Part__Feature006':('link5','wrist_connector'), 'joint6_rotor':('link6','output_disk'),
 'Part__Feature008':('gripper_base','gripper_drive_rail'),
 'Part__Feature015':('gripper_base','camera_base'),
 'Part__Feature009':('left_finger','left_slider'), 'Part__Feature010':('right_finger','right_slider'),
 'Part__Feature011':('left_finger','left_support'), 'Part__Feature012':('left_finger','left_soft_finger'),
 'Part__Feature013':('right_finger','right_support'), 'Part__Feature014':('right_finger','right_soft_finger')}
assert set(mapping)==set(byid)-{'Part__Feature007'},(set(mapping)^set(byid))
meshes={};inertial_parts={n:[] for n in ['gripper_base','left_finger','right_finger']};inventory=[]
for pid,(ln,an) in mapping.items():
 part=byid[pid];T=np.linalg.inv(Tcad[ln]);m=trimesh.load(CAD/part['mesh']);m.apply_scale(.001);m.apply_transform(T)
 color=[.64,.68,.72,1.]
 if 'soft_finger' in an:color=[1.,.67,.30,1.]
 elif 'support' in an:color=[.95,.95,.95,1.]
 elif 'camera' in an:color=[.22,.29,.34,1.]
 elif an=='gripper_drive_rail':color=[.784314,.784314,.784314,1.]
 elif 'slider' in an:color=[.37,.40,.44,1.]
 elif 'connector' in an:color=[.08,.09,.1,1.]
 fn='meshes/visual/'+an+'.stl';meshes[fn]=m
 for fmt in ('urdf','xml'):m.export(OUT/fmt/fn,file_type='stl')
 links[ln]['visuals'].append({'mesh':fn,'color':color})
 for k,s in enumerate(part['solids']):
  sm=trimesh.load(CAD/s['mesh']);sm.apply_scale(.001);sm.apply_transform(T)
  if sm.volume<1e-12 or len(sm.vertices)<4:continue
  hull=sm.convex_hull;cf='meshes/collision/'+an+f'_{k}.stl';meshes[cf]=hull
  for fmt in ('urdf','xml'):hull.export(OUT/fmt/cf,file_type='stl')
  links[ln]['collisions'].append({'mesh':cf})
  if ln in inertial_parts:
   volume=s['volume_mm3']*1e-9
   density=7800 if 'slider' in an else (1050 if 'soft_finger' in an else 1250)
   if an=='gripper_drive_rail':density=.45/(sum(v['volume_mm3'] for v in part['solids'])*1e-9)
   mass=volume*density;com=trimesh.transform_points([np.array(s['center_mm'])*.001],T)[0]
   I=np.array(s['inertia_mm5']).reshape(4,4)[:3,:3]*1e-15*density;I=T[:3,:3]@I@T[:3,:3].T
   inertial_parts[ln].append((mass,com,I))
 inventory.append({'cad_label':part['label'],'cad_id':pid,'link':ln,'visual_mesh':fn,'triangles':len(m.faces),'solid_count':len(part['solids'])})
for n,entries in inertial_parts.items():
 mass=sum(e[0] for e in entries);com=sum(e[0]*e[1] for e in entries)/mass;I=np.zeros((3,3))
 for ma,c,ii in entries:
  d=c-com;I+=ii+ma*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
 links[n]['inertial']={'mass':mass,'com':com.tolist(),'tensor':I.tolist(),'source':'Estimate: CAD mass distribution; drive/rail nominal 0.45 kg; supports 1250, silicone 1050, sliders 7800 kg/m^3. Not measured.'}
# Tool frame at midpoint between inner finger tips, following the wrist (not a measured TCP calibration).
lf=trimesh.load(CAD/'Part__Feature012.stl');rf=trimesh.load(CAD/'Part__Feature014.stl');min_gap=(rf.bounds[0,0]-lf.bounds[1,0])*.001-2*q_ref
axis6=np.array(cal['axis6']);all_tip=np.vstack([lf.vertices,rf.vertices])*.001;z=all_tip@axis6;tips=all_tip[z>z.max()-.005];tcp_cad=tips.mean(0);tcp_cad[0]=(lc[0]+rc[0])/2
tcp=Tc[6].copy();tcp[:3,3]=tcp_cad;toolT=np.linalg.inv(Tcad['gripper_base'])@tcp
links['tool0']={'name':'tool0','visuals':[],'collisions':[]};joints.append({'name':'tool0_fixed','type':'fixed','parent':'gripper_base','child':'tool0','T':toolT.tolist()})
# Use user-supplied material-separated arm meshes, preserving recovered diffuse colours.
material_map=json.loads((OUT/'metadata/textured_materials.json').read_text())['materials']
appearance_inventory=[]
for ln in names[:6]:
 links[ln]['visuals']=[]
 prefix='base_link' if ln=='base_link' else ln+'_rot'
 for path in sorted(TEXTURED.glob(prefix+'_*.obj')):
  mids=re.findall(r'^usemtl (.+)$',path.read_text(),re.M);assert len(mids)==1 and mids[0] in material_map,path
  color=material_map[mids[0]]['rgba'];mesh=trimesh.load(path,process=False,skip_materials=True,force='mesh')
  fn='meshes/visual/'+path.name;meshes[fn]=mesh
  obj=trimesh.exchange.obj.export_obj(mesh,include_normals=True,include_color=False,include_texture=False)
  for fmt in ('urdf','xml'):(OUT/fmt/fn).write_text(obj)
  links[ln]['visuals'].append({'mesh':fn,'color':color})
  appearance_inventory.append({'source':path.name,'source_material':mids[0],'rgba':color,'link':ln,'visual_mesh':fn,'triangles':len(mesh.faces)})
 assert links[ln]['visuals'],ln
# Stock output disk and stock assembly flange: geometry AND collisions replaced.
for ln,stem in [('link6','link6'),('flange_link','flange')]:
 mesh=trimesh.load(OFF/'piper/meshes'/(stem+'.stl'));fn='meshes/visual/native_'+stem+'.stl';meshes[fn]=mesh
 links[ln]['visuals']=[{'mesh':fn,'color':[.117647,.117647,.117647,1.]}]
 cf='meshes/collision/native_'+stem+'.stl';meshes[cf]=mesh.convex_hull;links[ln]['collisions']=[{'mesh':cf}]
 for fmt in ('urdf','xml'):
  mesh.export(OUT/fmt/fn,file_type='stl');meshes[cf].export(OUT/fmt/cf,file_type='stl')
 appearance_inventory.append({'source':'AgileX/'+stem+'.stl','link':ln,'visual_mesh':fn,'triangles':len(mesh.faces),'rgba':[.117647,.117647,.117647,1.]})
# Drop obsolete generated meshes so the folders contain only the current model assets.
used={a['mesh'] for l in links.values() for k in ['visuals','collisions'] for a in l[k]}
for fn in (set(meshes)|previous_meshes)-used:
 for fmt in ('urdf','xml'):
  obsolete=OUT/fmt/fn
  if obsolete.is_file():obsolete.unlink()
meshes={fn:m for fn,m in meshes.items() if fn in used}
inventory=[x for x in inventory if x['visual_mesh'] in used]
# Canonical link transforms at q=0, independent of the saved CAD posture.
Tzero={'base_link':np.eye(4)}
for j in joints:Tzero[j['child']]=Tzero[j['parent']]@np.array(j['T'])
# URDF --------------------------------------------------------------------
robot=ET.Element('robot',name='piper_umi');ET.SubElement(robot,'mujoco') # harmless extension, compiler below
ET.SubElement(robot.find('mujoco'),'compiler',balanceinertia='false',discardvisual='false',strippath='false')
for n,l in links.items():
 el=ET.SubElement(robot,'link',name=n)
 if 'inertial' in l:
  a=l['inertial'];I=np.array(a['tensor']);ie=ET.SubElement(el,'inertial');ET.SubElement(ie,'origin',xyz=vals(a['com']),rpy='0 0 0');ET.SubElement(ie,'mass',value=vals([a['mass']]));ET.SubElement(ie,'inertia',**{k:vals([v]) for k,v in dict(ixx=I[0,0],iyy=I[1,1],izz=I[2,2],ixy=I[0,1],ixz=I[0,2],iyz=I[1,2]).items()})
 for kind in ('visual','collision'):
  for k,a in enumerate(l[kind+'s']):
   e=ET.SubElement(el,kind,name=n+'_'+kind+'_'+str(k));g=ET.SubElement(e,'geometry');ET.SubElement(g,'mesh',filename=a['mesh'])
   if kind=='visual':mat=ET.SubElement(e,'material',name=Path(a['mesh']).stem+'_material');ET.SubElement(mat,'color',rgba=vals(a['color']))
for j in joints:
 el=ET.SubElement(robot,'joint',name=j['name'],type=j['type']);T=np.array(j['T']);ET.SubElement(el,'origin',xyz=vals(T[:3,3]),rpy=vals(Rot.from_matrix(T[:3,:3]).as_euler('xyz')));ET.SubElement(el,'parent',link=j['parent']);ET.SubElement(el,'child',link=j['child'])
 if j['type']!='fixed':
  ET.SubElement(el,'axis',xyz=vals(j['axis']));ET.SubElement(el,'limit',**{k:vals([v]) for k,v in j['limit'].items()});ET.SubElement(el,'dynamics',damping='0.2' if j['type']=='prismatic' else '0.5',friction='0.0')
 if 'mimic' in j:ET.SubElement(el,'mimic',**{k:str(v) for k,v in j['mimic'].items()})
ET.indent(robot);ET.ElementTree(robot).write(OUT/'urdf/piper_umi.urdf',encoding='utf-8',xml_declaration=True)
# MuJoCo ------------------------------------------------------------------
mj=ET.Element('mujoco',model='piper_umi');ET.SubElement(mj,'compiler',angle='radian',autolimits='true',inertiafromgeom='false',fusestatic='false');ET.SubElement(mj,'option',timestep='0.002',integrator='implicitfast',gravity='0 0 -9.81',iterations='100')
defs=ET.SubElement(mj,'default');ET.SubElement(defs,'joint',damping='1',armature='0.01');ET.SubElement(defs,'geom',friction='0.8 0.005 0.0001',solref='0.008 1',solimp='0.95 0.99 0.001')
asset=ET.SubElement(mj,'asset')
for fn in meshes:ET.SubElement(asset,'mesh',name=fn.replace('/','_').replace('.stl',''),file=fn)
world=ET.SubElement(mj,'worldbody');body_by={};parent_joint={j['child']:j for j in joints}
for n,l in links.items():
 j=parent_joint.get(n);parent=body_by[j['parent']] if j else world;T=np.array(j['T']) if j else np.eye(4)
 if n=='tool0':ET.SubElement(parent,'site',name='tool0',pos=vals(T[:3,3]),quat=vals(quat(T[:3,:3])),size='0.006',rgba='1 0.3 0.1 1');continue
 body=ET.SubElement(parent,'body',name=n,pos=vals(T[:3,3]),quat=vals(quat(T[:3,:3])));body_by[n]=body
 if 'inertial' in l:
  a=l['inertial'];I=np.array(a['tensor']);ET.SubElement(body,'inertial',mass=vals([a['mass']]),pos=vals(a['com']),fullinertia=vals([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
 if j and j['type']!='fixed':ET.SubElement(body,'joint',name=j['name'],type='hinge' if j['type']=='revolute' else 'slide',axis=vals(j['axis']),range=vals([j['limit']['lower'],j['limit']['upper']]),damping='2' if j['type']=='prismatic' else '1',armature='0.002' if j['type']=='prismatic' else '0.01')
 for k,a in enumerate(l['visuals']):ET.SubElement(body,'geom',name=n+'_visual_'+str(k),type='mesh',mesh=a['mesh'].replace('/','_').replace('.stl',''),rgba=vals(a['color']),contype='0',conaffinity='0',group='2',mass='0')
 for k,a in enumerate(l['collisions']):ET.SubElement(body,'geom',name=n+'_collision_'+str(k),type='mesh',mesh=a['mesh'].replace('/','_').replace('.stl',''),rgba='0.5 0.8 0.5 0.3',group='3',mass='0')
eq=ET.SubElement(mj,'equality');ET.SubElement(eq,'joint',name='parallel_gripper',joint1='gripper_right_joint',joint2='gripper_left_joint',polycoef='0 1 0 0 0',solref='0.004 1',solimp='0.99 0.999 0.0001')
contact=ET.SubElement(mj,'contact')
for j in joints:
 if j['child']!='tool0':ET.SubElement(contact,'exclude',body1=j['parent'],body2=j['child'])
# Adjacent CAD solids and the rotor-to-gripper fixed assembly share designed interfaces.
for a,b in [('link5','flange_link'),('link5','gripper_base'),('link6','gripper_base'),('link6','left_finger'),('link6','right_finger'),('flange_link','left_finger'),('flange_link','right_finger')]:ET.SubElement(contact,'exclude',body1=a,body2=b)
act=ET.SubElement(mj,'actuator')
for j in joints:
 if j['type']=='fixed' or 'mimic' in j:continue
 slide=j['type']=='prismatic';ET.SubElement(act,'position',name=j['name']+'_position',joint=j['name'],kp='1500' if slide else '1200',kv='30' if slide else '70',ctrllimited='true',ctrlrange=vals([j['limit']['lower'],j['limit']['upper']]),forcelimited='true',forcerange=vals([-j['limit']['effort'],j['limit']['effort']]))
key=ET.SubElement(mj,'keyframe');qhome=np.r_[qcad,q_ref,q_ref];chome=np.r_[qcad,q_ref];ET.SubElement(key,'key',name='cad_assembly',qpos=vals(qhome),ctrl=vals(chome));ET.SubElement(key,'key',name='closed',qpos='0 0 0 0 0 0 0 0',ctrl='0 0 0 0 0 0 0')
ET.indent(mj);ET.ElementTree(mj).write(OUT/'xml/piper_umi.xml',encoding='utf-8',xml_declaration=True)
# USD: embedded meshes, local materials, SI units, flat rigid bodies and physics joints.
stage=Usd.Stage.CreateNew(str(OUT/'usd/piper_umi.usd'));UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.);UsdPhysics.SetStageKilogramsPerUnit(stage,1.)
root=UsdGeom.Xform.Define(stage,'/PiperUMI');stage.SetDefaultPrim(root.GetPrim());root.GetPrim().SetDocumentation('Piper + UMI: supplied arm appearance, stock J6 disk/flange, custom CAD gripper; see README.md for sources, estimated inertias and limits.')
def gfq(R):
 q=quat(R);return Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:])))
def place(prim,T):
 x=UsdGeom.Xformable(prim);x.AddTranslateOp().Set(Gf.Vec3d(*map(float,T[:3,3])));x.AddOrientOp().Set(gfq(T[:3,:3]))
for n,l in links.items():
 path='/PiperUMI/'+n;x=UsdGeom.Xform.Define(stage,path);place(x.GetPrim(),Tzero[n])
 if n=='tool0':continue
 UsdPhysics.RigidBodyAPI.Apply(x.GetPrim());a=l['inertial'];mass=UsdPhysics.MassAPI.Apply(x.GetPrim());mass.CreateMassAttr(a['mass']);mass.CreateCenterOfMassAttr(Gf.Vec3f(*a['com']));ev,vec=np.linalg.eigh(a['tensor'])
 if np.linalg.det(vec)<0:vec[:,0]*=-1
 mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*ev));mass.CreatePrincipalAxesAttr(gfq(vec))
 # PhysX extension attributes remain readable even in plain OpenUSD.
 x.GetPrim().AddAppliedSchema('PhysxRigidBodyAPI');x.GetPrim().CreateAttribute('physxRigidBody:maxDepenetrationVelocity',Sdf.ValueTypeNames.Float).Set(1.)
 for kind in ('visual','collision'):
  for k,a in enumerate(l[kind+'s']):
   m=meshes[a['mesh']];mp=UsdGeom.Mesh.Define(stage,path+'/'+kind+'_'+str(k));mp.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.vertices,dtype=np.float32)));mp.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.faces),3,dtype=np.int32)));mp.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m.faces,dtype=np.int32).ravel()));mp.CreateSubdivisionSchemeAttr('none')
   mp.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.bounds,dtype=np.float32)))
   if kind=='collision':
    UsdPhysics.CollisionAPI.Apply(mp.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mp.GetPrim()).CreateApproximationAttr('convexHull');mp.CreatePurposeAttr('guide');mp.CreateVisibilityAttr('invisible')
   else:
    if a['mesh'].endswith('.stl'):
     # Preserve CAD planar faces and sharp edges; averaging normals across all
     # welded STL vertices makes brackets appear dented in RTX rendering.
     mp.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(np.asarray(m.face_normals,dtype=np.float32),3,axis=0)));mp.SetNormalsInterpolation('faceVarying')
    else:
     mp.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.vertex_normals,dtype=np.float32)));mp.SetNormalsInterpolation('vertex')
    mp.CreateDisplayColorAttr([Gf.Vec3f(*a['color'][:3])]);mat=UsdShade.Material.Define(stage,'/PiperUMI/Looks/'+Path(a['mesh']).stem);shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*a['color'][:3]));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.55);mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mp.GetPrim()).Bind(mat)
# tool0 follows gripper_base using hierarchy, no massless rigid body.
toolprim=stage.GetPrimAtPath('/PiperUMI/tool0');stage.RemovePrim(toolprim.GetPath());tx=UsdGeom.Xform.Define(stage,'/PiperUMI/gripper_base/tool0');place(tx.GetPrim(),toolT)
baseprim=stage.GetPrimAtPath('/PiperUMI/base_link');UsdPhysics.ArticulationRootAPI.Apply(baseprim);baseprim.AddAppliedSchema('PhysxArticulationAPI');baseprim.CreateAttribute('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool).Set(True);baseprim.CreateAttribute('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(32);baseprim.CreateAttribute('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(8)
fix=UsdPhysics.FixedJoint.Define(stage,'/PiperUMI/joints/world_fixed');fix.CreateBody1Rel().SetTargets(['/PiperUMI/base_link'])
for j in joints:
 if j['child']=='tool0':continue
 path='/PiperUMI/joints/'+j['name'];cls={'fixed':UsdPhysics.FixedJoint,'revolute':UsdPhysics.RevoluteJoint,'prismatic':UsdPhysics.PrismaticJoint}[j['type']];ju=cls.Define(stage,path);ju.CreateBody0Rel().SetTargets(['/PiperUMI/'+j['parent']]);ju.CreateBody1Rel().SetTargets(['/PiperUMI/'+j['child']]);T=np.array(j['T']);align=np.eye(3)
 if j['type']!='fixed':
  axis=np.array(j['axis']);z=np.array([0.,0,1]);cross=np.cross(z,axis);dot=np.dot(z,axis)
  if dot<-.999999:align=Rot.from_euler('x',np.pi).as_matrix()
  elif np.linalg.norm(cross)>1e-12:align=Rot.from_rotvec(cross/np.linalg.norm(cross)*np.arccos(np.clip(dot,-1,1))).as_matrix()
  ju.CreateAxisAttr('Z');factor=180/np.pi if j['type']=='revolute' else 1.;ju.CreateLowerLimitAttr(j['limit']['lower']*factor);ju.CreateUpperLimitAttr(j['limit']['upper']*factor)
 ju.CreateLocalPos0Attr(Gf.Vec3f(*T[:3,3]));ju.CreateLocalPos1Attr(Gf.Vec3f(0));ju.CreateLocalRot0Attr(gfq(T[:3,:3]@align));ju.CreateLocalRot1Attr(gfq(align));ju.CreateCollisionEnabledAttr(False)
 if j['type']!='fixed' and 'mimic' not in j:
  slide=j['type']=='prismatic';dr=UsdPhysics.DriveAPI.Apply(ju.GetPrim(),'linear' if slide else 'angular');dr.CreateTypeAttr('force');dr.CreateStiffnessAttr(1500 if slide else 1200*np.pi/180);dr.CreateDampingAttr(30 if slide else 70*np.pi/180);dr.CreateMaxForceAttr(j['limit']['effort']);dr.CreateTargetPositionAttr(0.)
 if 'mimic' in j:
  prim=ju.GetPrim();prim.AddAppliedSchema('PhysxMimicJointAPI:rotX');prim.CreateRelationship('physxMimicJoint:rotX:referenceJoint').SetTargets(['/PiperUMI/joints/'+j['mimic']['joint']]);prim.CreateAttribute('physxMimicJoint:rotX:gearing',Sdf.ValueTypeNames.Float).Set(-1.);prim.CreateAttribute('physxMimicJoint:rotX:offset',Sdf.ValueTypeNames.Float).Set(0.);prim.CreateAttribute('physxMimicJoint:rotX:referenceJointAxis',Sdf.ValueTypeNames.Token).Set('rotX')
# Explicit adjacent filtering matches the MuJoCo parent-child rule including rigid mount.
for j in joints:
 if j['child']=='tool0':continue
 api=UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath('/PiperUMI/'+j['parent']));api.CreateFilteredPairsRel().AddTarget('/PiperUMI/'+j['child'])
for a,b in [('link5','flange_link'),('link5','gripper_base'),('link6','gripper_base'),('link6','left_finger'),('link6','right_finger'),('flange_link','left_finger'),('flange_link','right_finger')]:UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath('/PiperUMI/'+a)).CreateFilteredPairsRel().AddTarget('/PiperUMI/'+b)
stage.GetRootLayer().Save()
spec={'robot':'piper_umi','links':links,'joints':joints,'cad_qpos':qhome.tolist(),'cad_ctrl':chome.tolist(),'gripper':{'coordinate':'single-finger outward travel in metres; opposite axes, equal coordinates','range_per_finger_m':[0,.05],'nominal_total_stroke_m':.1,'minimum_slider_clearance_m':.0002,'cad_q_per_finger_m':q_ref,'minimum_projected_inner_finger_gap_m':min_gap,'maximum_projected_inner_finger_gap_m':min_gap+.1,'note':'Minimum is CAD-derived conservative slider spacing; 100 mm stroke is nominal. Not a measured hardware calibration. Soft fingers modeled as rigid.'},'wrist_partition':raw['wrist_partition'],'part_inventory':inventory,'appearance_inventory':appearance_inventory,'source_link_frames_cad':{n:t.tolist() for n,t in Tcad.items()},'mount_revision':{'native_flange':'AgileX flange.stl','removed_cad_component':'UMI-PIPER-CAMERA-MOUNT','original_drive_origin_in_link6_m':cad_drive_origin.tolist(),'new_drive_origin_in_link6_m':[0,0,.0045],'eef_translation_in_link6_m':(np.array([0,0,.0045])-cad_drive_origin).tolist(),'retained_rail_length_m':.155},'official_source':'https://github.com/agilexrobotics/agx_arm_urdf','official_commit':'f6642ce0d7872c686f29c99e9e10cd23d1d49313'}
spec['appearance_colors']={'supports_rgba':[.95,.95,.95,1.],'fingers_rgba':[1.,.67,.30,1.],'finger_color_description':'light apricot-yellow'}
(OUT/'metadata/model_spec.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
for fmt in ('urdf','xml','usd'):
 shutil.copyfile(OFF/'LICENSE',OUT/fmt/'AGILEX_LICENSE.txt')
 (OUT/fmt/'model_info.json').write_text(json.dumps({k:spec[k] for k in ['robot','cad_qpos','cad_ctrl','gripper','mount_revision','appearance_inventory','appearance_colors','official_source','official_commit']},indent=2)+'\n')
print('Built',OUT,'visual meshes',sum(len(l['visuals']) for l in links.values()),'collision meshes',sum(len(l['collisions']) for l in links.values()),'gripper',spec['gripper'])

from build_scenes import build_scenes
build_scenes(OUT)
