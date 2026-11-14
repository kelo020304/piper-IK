#!/usr/bin/env python3
"""Build independent MuJoCo/Isaac preview scenes with blue checker floors."""
from pathlib import Path
import argparse,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,UsdLux,Sdf,Gf,Vt

def build_scenes(root):
 scene=ET.Element('mujoco',model='piper_umi_scene');ET.SubElement(scene,'include',file='piper_umi.xml')
 v=ET.SubElement(scene,'visual');ET.SubElement(v,'global',azimuth='135',elevation='-25',offwidth='1280',offheight='960')
 ET.SubElement(v,'headlight',diffuse='0.6 0.6 0.6',ambient='0.3 0.3 0.3',specular='0 0 0')
 ET.SubElement(scene,'statistic',center='0 0 0.2',extent='0.8')
 a=ET.SubElement(scene,'asset')
 ET.SubElement(a,'texture',type='skybox',builtin='gradient',rgb1='.3 .5 .7',rgb2='0 0 0',width='512',height='3072')
 ET.SubElement(a,'texture',name='blue_checker',type='2d',builtin='checker',mark='edge',rgb1='.2 .3 .4',rgb2='.1 .2 .3',markrgb='.8 .8 .8',width='256',height='256')
 ET.SubElement(a,'material',name='blue_floor',texture='blue_checker',texuniform='true',texrepeat='5 5',reflectance='.05')
 w=ET.SubElement(scene,'worldbody');ET.SubElement(w,'light',pos='1 -1 2',dir='-1 1 -2',diffuse='.8 .8 .8')
 ET.SubElement(w,'geom',name='floor',type='plane',size='0 0 .05',pos='0 0 -.002',material='blue_floor')
 ET.indent(scene);ET.ElementTree(scene).write(root/'xml/scene.xml',encoding='utf-8',xml_declaration=True)
 # Procedural checker asset, packaged locally with the USD scene.
 texdir=root/'usd/textures';texdir.mkdir(exist_ok=True)
 y,x=np.indices((256,256));colors=np.array([[.2,.3,.4],[.1,.2,.3]])
 data=(colors[((x//128+y//128)%2)]*255).astype(np.uint8)
 data[[0,-1],:]=204;data[:,[0,-1]]=204
 Image.fromarray(data).save(texdir/'blue_checker.png')
 path=root/'usd/scene.usda'
 if path.exists():path.unlink()
 st=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageUpAxis(st,'Z');UsdGeom.SetStageMetersPerUnit(st,1);UsdPhysics.SetStageKilogramsPerUnit(st,1)
 world=UsdGeom.Xform.Define(st,'/World');st.SetDefaultPrim(world.GetPrim())
 robot=st.DefinePrim('/World/PiperUMI','Xform');robot.GetReferences().AddReference('piper_umi.usd')
 ps=UsdPhysics.Scene.Define(st,'/World/PhysicsScene');ps.CreateGravityDirectionAttr((0,0,-1));ps.CreateGravityMagnitudeAttr(9.81)
 dome=UsdLux.DomeLight.Define(st,'/World/Light');dome.CreateIntensityAttr(300);dome.CreateColorAttr((1,1,1))
 key=UsdLux.DistantLight.Define(st,'/World/KeyLight');key.CreateIntensityAttr(1000);key.CreateAngleAttr(2);UsdGeom.Xformable(key).AddRotateXYZOp().Set((-30,-35,20))
 fill=UsdLux.DistantLight.Define(st,'/World/FillLight');fill.CreateIntensityAttr(700);fill.CreateAngleAttr(12);UsdGeom.Xformable(fill).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.8,-.8,1.2),Gf.Vec3d(0,0,.1),Gf.Vec3d(0,0,1)).GetInverse())
 floor=UsdGeom.Mesh.Define(st,'/World/Floor');floor.CreatePointsAttr([(-10,-10,-.002),(10,-10,-.002),(10,10,-.002),(-10,10,-.002)])
 floor.CreateFaceVertexCountsAttr([4]);floor.CreateFaceVertexIndicesAttr([0,1,2,3]);floor.CreateSubdivisionSchemeAttr('none')
 UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
 uv=UsdGeom.PrimvarsAPI(floor).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex');uv.Set([(0,0),(100,0),(100,100),(0,100)])
 mat=UsdShade.Material.Define(st,'/World/Looks/BlueChecker')
 shader=UsdShade.Shader.Define(st,str(mat.GetPath())+'/Surface');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.85)
 tex=UsdShade.Shader.Define(st,str(mat.GetPath())+'/Texture');tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('textures/blue_checker.png'));tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
 tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
 uvread=UsdShade.Shader.Define(st,str(mat.GetPath())+'/UV');uvread.CreateIdAttr('UsdPrimvarReader_float2');uvread.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');uvread.CreateOutput('result',Sdf.ValueTypeNames.Float2)
 tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uvread.ConnectableAPI(),'result')
 shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb');mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(mat)
 camera=UsdGeom.Camera.Define(st,'/World/PreviewCamera');camera.CreateFocalLengthAttr(40);camera.CreateHorizontalApertureAttr(36);camera.CreateClippingRangeAttr((.01,100))
 transform=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.85,-.85,.65),Gf.Vec3d(0,0,.18),Gf.Vec3d(0,0,1)).GetInverse();UsdGeom.Xformable(camera).AddTransformOp().Set(transform)
 st.GetRootLayer().Save()
 print('Built blue checker preview scenes')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);build_scenes(p.parse_args().root.resolve())
