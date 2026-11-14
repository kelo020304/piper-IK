"""The standalone pose filter (filters.py, piper-IK 0.2.0 unchanged): three cascaded stages, short-arc rotation,
input checks, speed caps; and the execution chain never calls it."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from piper_ik import ThirdOrderPoseFilter

PKG = Path(__file__).resolve().parents[1]


def test_filter_step_is_three_cascaded_stages_not_one():
    initial = np.eye(4); target = initial.copy(); target[0, 3] = 1
    f = ThirdOrderPoseFilter(initial, tau=.06)
    result = f.update(target, .02)
    alpha = 1 - np.exp(-.02 / .06)
    assert result[0, 3] == pytest.approx(alpha ** 3)
    result[:] = 0
    assert f.value[3, 3] == 1                                                    # value is a copy
    for _ in range(200):
        result = f.update(target, .02)
    np.testing.assert_allclose(result, target, atol=1e-12)
    f.reset(initial); np.testing.assert_array_equal(f.value, initial)


def test_filter_rotation_short_arc_and_constant_pose():
    a = np.eye(4); b = np.eye(4)
    a[:3, :3] = Rotation.from_euler('z', 179, degrees=True).as_matrix()
    b[:3, :3] = Rotation.from_euler('z', -179, degrees=True).as_matrix()
    f = ThirdOrderPoseFilter(a)
    for _ in range(50):
        out = f.update(b, .02)
        angle = Rotation.from_matrix(out[:3, :3] @ a[:3, :3].T).as_rotvec()[2]
        assert 0 <= angle <= np.deg2rad(2) + 1e-12
        np.testing.assert_allclose(out[:3, :3].T @ out[:3, :3], np.eye(3), atol=1e-12)
    f.reset(b); np.testing.assert_allclose(f.update(b, .01), b, atol=1e-12)


@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf])
def test_filter_invalid_time_does_not_change_state(value):
    f = ThirdOrderPoseFilter(np.eye(4))
    with pytest.raises(ValueError):
        f.update(np.eye(4), value)
    np.testing.assert_array_equal(f.value, np.eye(4))
    with pytest.raises(ValueError):
        ThirdOrderPoseFilter(np.eye(4), tau=value)


@pytest.mark.parametrize('bad', [np.zeros((4, 4)), np.diag([1., 1., -1., 1.]), np.full((4, 4), np.nan)])
def test_filter_refuses_invalid_poses(bad):
    with pytest.raises(ValueError):
        ThirdOrderPoseFilter(bad)


def test_filter_speed_caps():
    target = np.eye(4); target[0, 3] = 1
    target[:3, :3] = Rotation.from_euler('z', 1).as_matrix()
    f = ThirdOrderPoseFilter(np.eye(4), tau=.001, linear_speed=.02, angular_speed=.1)
    for _ in range(10):
        before = f.value; after = f.update(target, .02)
        assert np.linalg.norm(after[:3, 3] - before[:3, 3]) <= .02 * .02 + 1e-12
        assert Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).magnitude() <= .1 * .02 + 1e-12


def test_execution_chain_never_uses_the_filter():
    """AGENTS.md rule 2: core / realtime / trajectory / follow / arm / sim do not import filters.py."""
    for name in ('core.py', 'realtime.py', 'trajectory.py', 'follow.py', 'arm.py', 'sim.py', 'kinematics.py', 'convention.py'):
        tree = ast.parse((PKG / name).read_text())
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
                   {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any('filters' in m for m in imported), name
