# Piper + UMI robot models

[中文图文说明 / 最新运动预览](README.html)

- `urdf/piper_umi.urdf` — URDF, local visual/collision meshes, mimic joint, optional ROS 2 visualization package.
- `xml/piper_umi.xml` — MuJoCo MJCF, 7 actuators and gripper equality constraint; `xml/scene.xml` adds a demo environment.
- `usd/piper_umi.usd` — USD articulation with embedded geometry/materials and PhysX mimic; `usd/scene.usda` is a demo scene.

**Each format folder is self-contained and can be copied independently.** No runtime references to sibling folders, original CAD, development-machine paths or symlinks. Simulation programs are installed separately. Original CAD/archive files were not modified.

## Appearance and native mounting revision

The two UMI finger supports are white (RGBA 0.95, 0.95, 0.95, 1); the two fingers are light apricot-yellow (1, 0.67, 0.30, 1). Base through link5 use 76 supplied material-separated Piper OBJ meshes. Official DAE diffuse colors restore the archive's missing material library, including dark covers and contrasting label geometry. The robot needs no external image textures or MTL libraries. The USD demo scene includes its own local blue checker floor texture.

`link6` uses the stock output disk. A stock `flange_link` replaces the custom `UMI-PIPER-CAMERA-MOUNT`; its joint to link6 is fixed. The gripper drive is mounted at the official +4.5 mm offset. Relative to the previous custom mount, the retained end effector shifts approximately +3.013 mm along link6 Z, with minor XY centering corrections (0.087 mm and 0.0035 mm). Exact transforms are in `model_info.json`. The custom 155 mm rail, UMI finger spacing, 100 mm nominal total stroke and camera base are retained. The archive's different 145 mm stock gripper is not substituted.

MuJoCo and Isaac preview scenes use blue checker floors. Isaac screenshots: [overview](validation/isaac_sim_overview.png), [gripper close-up](validation/isaac_sim_gripper.png). The USD floor image is bundled under `usd/textures/`.

## Verification

All three folders were copied independently and loaded. URDF and MuJoCo FK were compared over 32 configurations. Eight retained custom CAD parts were checked in their source frames; the new stock mounting offset is intentional. All 86 visual colors match across URDF, MJCF and USD, with no unresolved OBJ material libraries. MuJoCo completed 5500 gravity/collision-enabled steps; Isaac Sim 5.1.0 loaded the articulation and passed FK and physical gripper mimic/full-stroke checks with gravity and collisions. See `validation/checks.json` and `validation/isaac_sim.json`. ROS 2 launch was not executed. These tests establish model consistency, not measured hardware calibration or a general contact benchmark.

Soft fingers are rigid; custom gripper masses/inertias are estimates. Arm collision hulls retain the previous calibrated CAD approximation, rather than reproducing every detail of the replacement visual meshes. See each folder's README for limits and control conventions.

## Rebuild

Dependencies: FreeCAD (tested 0.18.4) for STEP extraction; scientific Python, trimesh, MuJoCo and OpenUSD for generation/validation (`scripts/requirements.txt`). Supply the original STEP and extracted appearance archive. Cached source colors are already in `metadata/textured_materials.json`. `metadata/cad_alignment.json` applies to this exact STEP; review calibration and grouping if the assembly changes.

```bash
# From this repository root:
python scripts/extract_step.py /path/to/PIPER-UMI-GRIPPER.STEP _build/cad
unzip /path/to/piper_textured_meshes.zip -d _build/textured

git clone https://github.com/agilexrobotics/agx_arm_urdf.git _build/agx_arm_urdf
git -C _build/agx_arm_urdf checkout f6642ce0d7872c686f29c99e9e10cd23d1d49313
python scripts/build_models.py --cad-dir _build/cad --official-dir _build/agx_arm_urdf --textured-dir _build/textured/piper_textured_meshes --out .
python scripts/validate_models.py --cad-dir _build/cad
# Using Isaac Sim's Python:
python scripts/validate_isaac.py
# Optional PNG/MP4 previews:
python scripts/render_preview.py
# Actual Isaac Sim RTX screenshots (Isaac Sim Python):
python scripts/render_isaac_preview.py
```

`metadata/model_spec.json` records link inertias, joints, source grouping, mounting correction and gripper assumptions. `metadata/source_manifest.json` records source hashes and the upstream version. Original STEP/ZIP are rebuild inputs and are not runtime dependencies.

## Repository contents

Use this directory as the model repository root. `.gitignore` excludes build/cache files. All current assets are below 100 MB each; Git LFS is not required for this snapshot. Preserve the complete format folders. After rebuilding, rerun validation and refresh bundled reports/checksums before release. No git push was performed.

See `THIRD_PARTY_NOTICES.md` for license scope. Upstream license texts are included in each independent folder. User-provided CAD/mesh assets have not been relicensed.
