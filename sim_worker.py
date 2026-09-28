"""Live native simulator, main-thread graphics, private local socket. Kinematic IK."""
import argparse,base64,io,json,os,select,signal,socket,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
NAMES=['base_link']+[f'link{i}' for i in range(1,7)]
def matrix(p,q):
 T=np.eye(4);T[:3,3]=p;T[:3,:3]=Rotation.from_quat(q).as_matrix();return T
class MuJoCoSim:
 def __init__(self,headless):
  if headless:os.environ['MUJOCO_GL']='egl'
  import mujoco
  self.mj=mujoco;self.m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/models/piper_scene.xml'));self.d=mujoco.MjData(self.m)
  self.m.vis.headlight.ambient[:]=.55;self.m.vis.headlight.diffuse[:]=.8;self.renderer=None;self.viewer=None;self.target=np.eye(4);self.tcp=np.eye(4);self.frames=0
  if not headless:
   import mujoco.viewer
   self.viewer=mujoco.viewer.launch_passive(self.m,self.d,show_left_ui=False,show_right_ui=False)
   self.viewer.cam.lookat[:]=[.12,0,.32];self.viewer.cam.distance=1.5;self.viewer.cam.azimuth=135;self.viewer.cam.elevation=-24
 def state(self):
  links=[]
  for name in NAMES:
   i=self.m.body(name).id;links.append({'name':name,'position':self.d.xpos[i].tolist(),'quaternion':self.d.xquat[i][[1,2,3,0]].tolist()})
  i=self.m.body('tool0').id
  return {'q':self.d.qpos.tolist(),'flange':matrix(self.d.xpos[i],self.d.xquat[i][[1,2,3,0]]).tolist(),'links':links,'frames':self.frames,'backend':'mujoco'}
 def step(self,q,target,tcp):
  self.d.qpos[:]=q;self.d.qvel[:]=0;self.mj.mj_forward(self.m,self.d);self.target=target;self.tcp=tcp;self.frames+=1;return self.state()
 def axes(self,scene,T,length,dashed=False):
  for k,c in enumerate([[1,.1,.1,1],[.1,1,.3,1],[.1,.4,1,1]]):
   for a,b in ([(0,.2),(.3,.5),(.6,.8),(.9,1)] if dashed else [(0,1)]):
    g=scene.geoms[scene.ngeom];self.mj.mjv_initGeom(g,self.mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array(c));self.mj.mjv_connector(g,self.mj.mjtGeom.mjGEOM_CAPSULE,.0025,T[:3,3]+T[:3,k]*length*a,T[:3,3]+T[:3,k]*length*b);scene.ngeom+=1
 def pose_label(self,scene,T,text,offset,color):
  g=scene.geoms[scene.ngeom]
  self.mj.mjv_initGeom(g,self.mj.mjtGeom.mjGEOM_LABEL,np.zeros(3),T[:3,3]+T[:3,2]*offset,np.eye(3).flatten(),np.array(color))
  g.label=text;scene.ngeom+=1
 def add_axes(self,scene):
  self.axes(scene,np.eye(4),.09);self.axes(scene,np.array(self.state()['flange'])@self.tcp,.08);self.axes(scene,self.target,.13,True)
  self.pose_label(scene,self.target,'TARGET',.17,[1.,.76,.4,1.])
  self.pose_label(scene,np.array(self.state()['flange'])@self.tcp,'ACTUAL',-.055,[.9,.96,1.,1.])
 def render(self):
  if self.viewer:
   if not self.viewer.is_running():return False
   with self.viewer.lock():self.viewer.user_scn.ngeom=0;self.add_axes(self.viewer.user_scn)
   self.viewer.sync()
  return True
 def capture(self):
  if self.renderer is None:self.renderer=self.mj.Renderer(self.m,height=480,width=640)
  cam=self.mj.MjvCamera();cam.lookat[:]=[.12,0,.32];cam.distance=1.5;cam.azimuth=135;cam.elevation=-24
  self.renderer.update_scene(self.d,camera=cam);self.add_axes(self.renderer.scene);return self.renderer.render()
 def close(self):
  if self.viewer:self.viewer.close()
  if self.renderer:self.renderer.close()
class IsaacSim:
 def __init__(self,headless):
  from isaacsim import SimulationApp
  self.app=SimulationApp({'headless':headless,'renderer':'RaytracedLighting','width':960,'height':720,'multi_gpu':False,'limit_cpu_threads':4,'anti_aliasing':0})
  import omni.kit.commands,omni.usd
  from pxr import UsdPhysics,PhysxSchema,UsdLux,UsdGeom,Gf
  from isaacsim.core.api import World
  from isaacsim.core.prims import Articulation
  self.Gf=Gf;self.UsdGeom=UsdGeom
  self.world=World(stage_units_in_meters=1.,physics_dt=.02,rendering_dt=1/30,backend='numpy',device='cpu')
  _,cfg=omni.kit.commands.execute('URDFCreateImportConfig');cfg.merge_fixed_joints=False;cfg.fix_base=True;cfg.import_inertia_tensor=True;cfg.convex_decomp=False
  ok,root=omni.kit.commands.execute('URDFParseAndImportFile',urdf_path=str(ROOT/'assets/models/piper.urdf'),import_config=cfg,get_articulation_root=True)
  if not ok:raise RuntimeError('Piper URDF import failed')
  self.stage=omni.usd.get_context().get_stage()
  for prim in self.stage.Traverse():
   if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
   if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
  light=UsdLux.DomeLight.Define(self.stage,'/World/Light');light.CreateIntensityAttr(1200)
  self.art=self.world.scene.add(Articulation(prim_paths_expr=root,name='piper'));self.world.reset();self.art.set_gains(kps=np.zeros((1,6)),kds=np.zeros((1,6)))
  self.order=[self.art.get_dof_index(f'joint{i}') for i in range(1,7)];self.indices={name:self.art.get_body_index(name) for name in NAMES};self.tip=self.art.get_body_index('tool0');self.frames=0
  self.base_axes=self.make_axes('BaseAxes',.09);self.actual_axes=self.make_axes('ActualAxes',.08);self.target_axes=self.make_axes('TargetAxes',.13,True)
  cam=UsdGeom.Camera.Define(self.stage,'/World/PiperCamera');view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(-1.3,1.3,1.0),Gf.Vec3d(.12,0,.34),Gf.Vec3d(0,0,1));cam.AddTransformOp().Set(view.GetInverse());cam.CreateFocalLengthAttr(24.)
  # Two meshes keep each checker color uniform across renderers. Visual only.
  UsdGeom.Xform.Define(self.stage,'/World/Floor')
  for parity,color in enumerate([(.1,.2,.3),(.2,.3,.4)]):
   points=[]
   for ix in range(20):
    for iy in range(20):
     if (ix+iy)%2!=parity:continue
     x=-2+ix*.2;y=-2+iy*.2
     points.extend([(x,y,-.01),(x+.2,y,-.01),(x+.2,y+.2,-.01),(x,y+.2,-.01)])
   floor=UsdGeom.Mesh.Define(self.stage,f'/World/Floor/Blue{parity}')
   floor.CreatePointsAttr(points);floor.CreateFaceVertexCountsAttr([4]*(len(points)//4));floor.CreateFaceVertexIndicesAttr(list(range(len(points))));floor.CreateDisplayColorAttr([color]);floor.CreateSubdivisionSchemeAttr('none')
  if not headless:
   from omni.kit.viewport.utility import get_active_viewport
   get_active_viewport().camera_path='/World/PiperCamera'
  self.product=None;self.rgb=None
 def make_axes(self,name,length,dashed=False):
  U=self.UsdGeom;G=self.Gf;parent=U.Xform.Define(self.stage,'/World/'+name);op=parent.AddTransformOp()
  for axis,color in enumerate([(1.,.1,.1),(.1,1.,.3),(.1,.4,1.)]):
   for j,(a,b) in enumerate([(0,.2),(.3,.5),(.6,.8),(.9,1)] if dashed else [(0,1)]):
    c=U.Cylinder.Define(self.stage,f'/World/{name}/a{axis}_{j}');c.CreateAxisAttr(['X','Y','Z'][axis]);c.CreateRadiusAttr(.0025);c.CreateHeightAttr(length*(b-a));c.CreateDisplayColorAttr([color]);xyz=[0.,0.,0.];xyz[axis]=length*(a+b)/2;c.AddTranslateOp().Set(G.Vec3d(*xyz))
  return op
 def state(self):
  trans=np.asarray(self.art._physics_view.get_link_transforms())[0]
  links=[{'name':name,'position':trans[i,:3].tolist(),'quaternion':trans[i,3:7].tolist()} for name,i in self.indices.items()]
  return {'q':np.asarray(self.art.get_joint_positions())[0,self.order].tolist(),'flange':matrix(trans[self.tip,:3],trans[self.tip,3:7]).tolist(),'links':links,'frames':self.frames,'backend':'isaac'}
 def step(self,q,target,tcp):
  positions=np.zeros((1,6),np.float32);positions[0,self.order]=q;self.art.set_joint_positions(positions);self.art.set_joint_velocities(np.zeros((1,6),np.float32));self.world.step(render=False);self.frames+=1
  state=self.state();self.actual_axes.Set(self.Gf.Matrix4d((np.array(state['flange'])@tcp).T.tolist()));self.target_axes.Set(self.Gf.Matrix4d(target.T.tolist()));return state
 def render(self):
  if not self.app.is_running():return False
  self.world.render();return True
 def capture(self):
  if self.product is None:
   import omni.replicator.core as rep
   self.product=rep.create.render_product('/World/PiperCamera',(640,480));self.rgb=rep.AnnotatorRegistry.get_annotator('rgb');self.rgb.attach(self.product)
  for _ in range(8):self.world.render()
  frame=self.rgb.get_data();assert frame.shape[:2]==(480,640);return frame[:,:,:3]
 def close(self):self.app.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--sim',choices=['mujoco','isaac'],required=True);p.add_argument('--socket',required=True);p.add_argument('--headless',action='store_true');a=p.parse_args();sys.argv=[sys.argv[0]]
 running=True
 def stop(*_):
  nonlocal running
  running=False
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 sim=None;listener=None;conn=None
 try:
  sim=MuJoCoSim(a.headless) if a.sim=='mujoco' else IsaacSim(a.headless)
  sim.step(np.array([0,1,-1.2,.2,.55,0]),np.eye(4),np.eye(4))
  listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(a.socket);os.chmod(a.socket,0o600);listener.listen(1);listener.setblocking(False)
  print('SIM_READY',a.sim,flush=True);buf=b'';last_render=0
  while running:
   ready,_,_=select.select([conn or listener],[],[],.005)
   if ready:
    if conn is None:conn,_=listener.accept();conn.settimeout(2)
    else:
     block=conn.recv(65536)
     if not block:break
     buf+=block
     if len(buf)>1024*1024:raise ValueError('Oversized simulator request')
     while b'\n' in buf:
      line,buf=buf.split(b'\n',1)
      try:
       msg=json.loads(line);op=msg.get('op')
       if op=='quit':running=False;break
       if op=='ping':answer=sim.state()
       elif op=='step':
        q=np.array(msg['q']);target=np.array(msg['target']);tcp=np.array(msg['tcp'])
        if q.shape!=(6,) or target.shape!=(4,4) or tcp.shape!=(4,4) or not all(np.isfinite(x).all() for x in [q,target,tcp]):raise ValueError('Invalid simulator command')
        answer=sim.step(q,target,tcp)
       elif op=='capture':
        from PIL import Image
        f=io.BytesIO();Image.fromarray(sim.capture()).save(f,format='PNG');answer={'png':base64.b64encode(f.getvalue()).decode()}
       else:raise ValueError('Unknown simulator operation')
      except Exception as exc:answer={'error':str(exc)}
      conn.sendall((json.dumps(answer)+'\n').encode())
   if time.monotonic()-last_render>=1/30:
    if not sim.render():break
    last_render=time.monotonic()
 finally:
  if conn:conn.close()
  if listener:listener.close()
  if sim:sim.close()
if __name__=='__main__':main()
