"""The bundled models: every file of the Piper + UMI model package matches its SHA256SUMS, every mesh the IK URDF
names exists, and model/tcp.json is the transform the console derives from the bundled MuJoCo XML."""
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest
from piper_ik import PiperIK, MODEL

MODELS = MODEL.parent / 'piper_umi_models'


def test_model_package_complete_and_unchanged():
    lines = (MODELS / 'SHA256SUMS').read_text().splitlines()
    for line in lines:
        digest, name = line.split(maxsplit=1); name = name.lstrip('*')
        assert hashlib.sha256((MODELS / name).read_bytes()).hexdigest() == digest, name
    for f in ('urdf/piper_umi.urdf', 'xml/piper_umi.xml', 'xml/scene.xml', 'usd/piper_umi.usd', 'usd/scene.usda'):
        assert (MODELS / f).is_file(), f
    assert len(lines) > 300


def test_every_mesh_of_the_ik_urdf_exists():
    meshes = [m.get('filename') for m in ET.parse(MODEL).getroot().iter('mesh')]
    assert len(meshes) > 100 and all((MODEL.parent / m).is_file() for m in meshes)


def test_tcp_is_the_bundled_xml_tool0():
    """Same derivation as the console (data_viewer runtime_motion.make_ik): XML tool0 against the IK model's flange."""
    mujoco = pytest.importorskip('mujoco')
    m = mujoco.MjModel.from_xml_path(str(MODELS / 'xml/piper_umi.xml')); d = mujoco.MjData(m); ik = PiperIK()
    qid = [m.jnt_qposadr[m.joint(f'joint{i}').id] for i in range(1, 7)]; site = m.site('tool0').id
    for q in (np.array([0., .8, -1.2, 0., .5, 0.]), np.array([.2, .5, -1., .3, .4, .1]), np.zeros(6)):
        d.qpos[:] = 0; d.qpos[qid] = q; mujoco.mj_forward(m, d)
        T = np.eye(4); T[:3, :3] = d.site_xmat[site].reshape(3, 3); T[:3, 3] = d.site_xpos[site]
        flange = ik.fk(q) @ np.linalg.inv(ik.tcp)
        assert np.abs(np.linalg.inv(flange) @ T - ik.tcp).max() < 2e-4             # the console's own consistency bound
