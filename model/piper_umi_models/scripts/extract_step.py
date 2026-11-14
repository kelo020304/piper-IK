#!/usr/bin/env python3
"""FreeCAD Python: extract the supplied STEP assembly in world millimetres.
Usage: python3 extract_step.py ASSEMBLY.STEP OUTPUT_DIR
Needs FreeCAD/Part/Import/Mesh; no GUI and does not modify the input.
"""
import sys,json
from pathlib import Path
try: import FreeCAD as App
except ImportError:
 sys.path.append('/usr/lib/freecad/lib')
 import FreeCAD as App
import Part,Import,Mesh
source=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
doc=App.newDocument('PiperExport');Import.insert(str(source),doc.Name)
parts=[]
for obj in doc.Objects:
 if not hasattr(obj,'Shape') or obj.Shape.isNull():continue
 s=obj.Shape.copy();s.Placement=obj.getGlobalPlacement()
 parts.append([obj.Name,obj.Label,s])
# J52 contains the J6 output disk in one solid. Separate only its CAD material.
wrist=next(x for x in parts if x[1]=='J52.STEP');s=wrist[2]
cyl=next(f.Surface for f in s.Faces if isinstance(f.Surface,Part.Cylinder) and abs(f.Surface.Radius-17.5)<1e-6)
axis=cyl.Axis;center=cyl.Center;tip=center+axis # CAD mounting plane at cylinder parameter +1 mm
cutter=Part.makeCylinder(17.50001,4.00002,tip-axis*4,axis)
rotor=s.common(cutter);housing=s.cut(cutter)
assert not rotor.isNull() and len(rotor.Solids)>0
split={'source_volume_mm3':s.Volume,'housing_volume_mm3':housing.Volume,'rotor_volume_mm3':rotor.Volume,'volume_difference_mm3':s.Volume-housing.Volume-rotor.Volume,'rotor_radius_mm':17.5,'tip_cad_mm':list(tip),'axis_cad':list(axis)}
wrist[2]=housing;parts.append(['joint6_rotor','J6 output disk (partition of J52.STEP)',rotor])
result=[]
for name,label,s in parts:
 def export(shape,path):
  verts,faces=shape.tessellate(.15)
  mesh=Mesh.Mesh([[verts[i] for i in f] for f in faces]);mesh.write(str(path));return mesh.CountFacets
 facets=export(s,out/(name+'.stl'));solids=[]
 for k,solid in enumerate(s.Solids):
  fn=name+'_solid_'+str(k)+'.stl';export(solid,out/fn)
  solids.append({'mesh':fn,'volume_mm3':solid.Volume,'center_mm':list(solid.CenterOfMass),'inertia_mm5':list(solid.MatrixOfInertia.A)})
 result.append({'name':name,'label':label,'mesh':name+'.stl','triangles':facets,'solids':solids})
 print(name,label,facets,flush=True)
(out/'parts.json').write_text(json.dumps({'parts':result,'wrist_partition':split},ensure_ascii=False,indent=2)+'\n')
