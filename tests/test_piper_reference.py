"""The Piper chain gives the same numbers as piper_ik 1.0.0, the version compared cycle by cycle with the data_viewer
console (difference 0). tests/data/piper_v1_reference.npz was recorded with 1.0.0 on 2026-11-13: 120 IK solves (every
fourth target out of reach) and 1200 cycles (6 s) of the dry-run loop in 'wait' and 'strict' mode. Exact equality."""
from pathlib import Path
import numpy as np
import pytest
from piper_ik import PiperIK
from piper_ik.sim import SimArm

REF = np.load(Path(__file__).resolve().parent / 'data' / 'piper_v1_reference.npz')
Q_START = np.array([0., 1.0, -1.2, 0., .6, 0.])


def path(t, t_end=8.):
    t = float(np.clip(t, 0., t_end)); s = np.sin(np.pi * t / t_end)
    g = 1. if t < 3. else (0. if t < 6. else 1.)
    return np.array([.05 * s, .04 * np.sin(2 * np.pi * t / t_end), -.03 * s, 0., np.deg2rad(15) * s, 0., g])


def test_ik_solutions_unchanged():
    ik = PiperIK()
    for T, seed, q in zip(REF['ik_T'], REF['ik_seed'], REF['ik_q']):
        assert np.array_equal(ik.solve(T, seed).q, q)


@pytest.mark.parametrize('mode', ['wait', 'strict'])
def test_commands_unchanged(mode):
    """Chunks of 16 every 4 frames, an acceleration-limited arm, 5 ms cycles: joints, width and data time per cycle."""
    arm = SimArm(Q_START); ik = PiperIK(follow=mode); ik.start(arm.joints(), arm.gripper_width()); nxt = 0.; rows = []
    for _ in range(len(REF[f'run_{mode}'])):
        if ik.source_time >= nxt and ik.source_time < 8.:
            _, t_obs = ik.observation(arm.joints(), arm.gripper_width())
            ik.push_actions(np.array([path(t_obs + (k + 1) / ik.fps) for k in range(16)]), t_obs); nxt = t_obs + 4 / ik.fps
        q_cmd, width = ik.tick(arm.joints(), .005)
        arm.send_joints(q_cmd); arm.send_gripper(width); arm.step(.005)
        rows.append(np.r_[q_cmd, width, ik.source_time])
    assert np.array_equal(np.array(rows), REF[f'run_{mode}'])
