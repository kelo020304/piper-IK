"""Public coordinate contract, nonzero TCP and training/runtime policy parity."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from piper_ik.engine import Engine

ROOT = Path(__file__).resolve().parents[1]


def engine():
    config = json.loads((ROOT/'config.json').read_text())
    config['tcp_xyz_m'] = [.02, -.01, .08]
    config['tcp_rpy_deg'] = [5., -10., 20.]
    result = Engine(config)
    result.control('test', {'type': 'claim'})
    return result


def test_base_nudge_accumulates_absolute_target_and_left_multiplies_rotation():
    e = engine()
    before = e.target.copy()
    e.control('test', {'type': 'nudge', 'key': 'a'})
    np.testing.assert_allclose(e.target[:3, 3], before[:3, 3]+[0, .001, 0])
    e.last_nudge = -1e10
    e.control('test', {'type': 'nudge', 'key': 'q'})
    np.testing.assert_allclose(e.target[:3, :3], Rotation.from_rotvec([0, 0, np.deg2rad(.5)]).as_matrix()@before[:3, :3])


def test_absolute_pose_replaces_target_and_tcp_conversion_reaches_ik():
    e = engine()
    wanted = e.pose(e.q)
    wanted[:3, 3] += [.003, -.002, .001]
    message = {'type': 'pose_target', 'frame': 'base', 'position_m': wanted[:3, 3].tolist(),
               'quaternion_xyzw': Rotation.from_matrix(wanted[:3, :3]).as_quat().tolist()}
    e.control('test', message)
    e.control('test', message)
    np.testing.assert_allclose(e.target, wanted, atol=1e-12)
    captured = []
    solve = e.pin.solve
    def record(target, *args, **kwargs):
        captured.append(target.copy())
        return solve(target, *args, **kwargs)
    e.pin.solve = record
    e.tick()
    assert captured and e.fault is None
    np.testing.assert_allclose(captured[0]@e.tcp, e.reference_pose, atol=1e-12)


def test_centered_training_actor_matches_runtime_quaternion_and_delta_contract():
    spec = importlib.util.spec_from_file_location('standalone_training', ROOT/'training/train_rl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # No Warp / GPU required to inspect actor.
    e = engine()
    checkpoint = torch.load(ROOT/'assets/weights/ppo_precision.pt', map_location='cpu', weights_only=False)
    actor = module.ActorCritic(centered=True)
    actor.load_state_dict(checkpoint['model'])
    T = e.flange(e.q)
    G = T.copy()
    G[:3, 3] += [.004, -.006, .002]
    G[:3, :3] = Rotation.from_rotvec([.01, -.03, .02]).as_matrix()@T[:3, :3]
    def wxyz(matrix):
        q = np.roll(Rotation.from_matrix(matrix).as_quat(), 1)
        return q if q[0] >= 0 else -q
    obs = np.r_[T[:3, 3], wxyz(T[:3, :3]), G[:3, 3], wxyz(G[:3, :3]), e.q,
                (G[:3, 3]-T[:3, 3])*20,
                Rotation.from_matrix(G[:3, :3]@T[:3, :3].T).as_rotvec()*3].astype('float32')
    with torch.no_grad():
        delta = actor.mean(torch.from_numpy(obs)[None])[0].clamp(-1, 1).numpy()*.2
    expected = np.clip(e.q+delta, e.lo, e.hi)
    np.testing.assert_allclose(e.rl.propose(e.q, T, G), expected, atol=2e-7)
