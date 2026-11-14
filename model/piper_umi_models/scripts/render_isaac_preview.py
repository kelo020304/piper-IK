#!/usr/bin/env python3
"""Capture actual Isaac Sim RTX images of the packaged scene (no image retouching)."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args,_=p.parse_known_args();root=args.root.resolve()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','width':1600,'height':1200,'anti_aliasing':3,'multi_gpu':False,'limit_cpu_threads':4,'hide_ui':True})
import numpy as np
from PIL import Image
import carb,omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom,Gf
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
omni.usd.get_context().open_stage(str(root/'usd/scene.usda'))
for _ in range(20):app.update()
world=World(stage_units_in_meters=1.,physics_dt=1/240,rendering_dt=1/60,backend='numpy',device='cpu')
stage=omni.usd.get_context().get_stage();spec=json.loads((root/'metadata/model_spec.json').read_text())
art=world.scene.add(Articulation(prim_paths_expr='/World/PiperUMI/base_link',name='piper_umi'))
world.reset()
physical=[j for j in spec['joints'] if j['type']!='fixed'];order=[art.get_dof_index(j['name']) for j in physical]
positions=np.zeros((1,8),dtype=np.float32);positions[0,order]=spec['cad_qpos']
art.set_joint_positions(positions);art.set_joint_velocities(np.zeros_like(positions));art.set_joint_position_targets(positions)
# Hold the recorded assembly pose, allowing real scene lighting/materials to converge.
for _ in range(12):world.step(render=True)
trans=np.asarray(art._physics_view.get_link_transforms())[0]
target=np.mean([trans[art.get_body_index(n),:3] for n in ['left_finger','right_finger']],axis=0)
cam=UsdGeom.Camera(stage.GetPrimAtPath('/World/PreviewCamera'));cam.CreateVerticalApertureAttr(27)
xform=UsdGeom.Xformable(cam);xform.ClearXformOpOrder();op=xform.AddTransformOp()
rp=rep.create.render_product(str(cam.GetPath()),(1600,1200));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(rp)
rep.orchestrator.set_capture_on_play(False)
settings=carb.settings.get_settings()
print('TONEMAP',settings.get('/rtx/post/tonemap/op'),flush=True)
views=[('isaac_sim_overview',np.array([.72,-.72,.57]),np.array([0.,0.,.18])),('isaac_sim_gripper',target+np.array([.33,-.38,.26]),target+np.array([0.,0.,.025]))]
metadata={'runtime':'Isaac Sim 5.1.0','renderer':'RaytracedLighting','scene':'usd/scene.usda','resolution':[1600,1200],'postprocessed':False,'fingers_rgba':[1.,.67,.30,1.],'views':[]}
for name,eye,look in views:
 op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*look),Gf.Vec3d(0,0,1)).GetInverse())
 for _ in range(8):world.render()
 rep.orchestrator.step(rt_subframes=64,delta_time=0.,pause_timeline=False)
 rgba=np.asarray(rgb.get_data());assert rgba.shape==(1200,1600,4),rgba.shape
 assert rgba[:,:,:3].std()>8, 'Render is blank or overexposed'
 Image.fromarray(rgba).save(root/'validation'/(name+'.png'))
 metadata['views'].append({'image':name+'.png','eye':eye.tolist(),'target':look.tolist()})
 print('CAPTURED',name,rgba.shape,flush=True)
(root/'validation/isaac_render.json').write_text(json.dumps(metadata,indent=2)+'\n')
rgb.detach(rp);rp.destroy();app.close()
