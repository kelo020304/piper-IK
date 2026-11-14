#!/usr/bin/env python3
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,math
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args();root=args.root.resolve();out=root/'validation'
m=mujoco.MjModel.from_xml_path(str(root/'xml/scene.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.27];cam.distance=1.05;cam.azimuth=135;cam.elevation=-22
opt=mujoco.MjvOption();opt.geomgroup[3]=0
renderer=mujoco.Renderer(m,height=720,width=960)
def render():
 mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam,scene_option=opt);return renderer.render()
Image.fromarray(render()).save(out/'cad_pose.png')
for name,qg in [('gripper_closed',0.),('gripper_open',.05)]:
 d.qpos[6:]=qg;Image.fromarray(render()).save(out/(name+'.png'))
mujoco.mj_resetDataKeyframe(m,d,0)
with imageio.get_writer(out/'motion_preview.mp4',fps=30,codec='libx264',quality=7,macro_block_size=16) as writer:
 for k in range(240):
  t=k/30;blend=min(1,t)
  d.ctrl[:6]=np.array([.22*math.sin(.7*t),.35+.25*math.sin(.6*t),-.65-.25*math.sin(.6*t),.2*math.sin(.8*t),.2*math.cos(.7*t),.25*math.sin(.5*t)])*blend
  d.ctrl[6]=.025+.024*math.sin(1.5*t)
  for _ in range(17):mujoco.mj_step(m,d)
  writer.append_data(render())
assert np.isfinite(d.qpos).all() and not np.any(d.warning.number)
renderer.close();print('Wrote images and 8 s video')
