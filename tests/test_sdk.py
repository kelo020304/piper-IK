import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from piper_ik import PiperIK, PinocchioConfig, RLConfig, ThirdOrderPoseFilter

Q = np.array([0., 1., -1.2, .2, .55, 0.])
ROOT = Path(__file__).resolve().parents[1]


def test_filter_import_does_not_load_backend_dependencies():
    subprocess.run([sys.executable, '-c', "from piper_ik import ThirdOrderPoseFilter; import sys; assert 'torch' not in sys.modules and 'pinocchio' not in sys.modules"], check=True)


@pytest.mark.parametrize('backend', ['pinocchio', 'rl'])
def test_fk_roundtrip_and_finite_absolute_output(backend):
    if backend == 'rl':
        pytest.importorskip('torch')
    ik = PiperIK(backend)
    result = ik.solve(ik.fk(Q), Q)
    assert result.converged
    np.testing.assert_allclose(result.q, Q, atol=1e-12)
    assert result.position_error < 1e-12


def test_pin_local_target_and_nonzero_tcp():
    tcp = np.eye(4)
    tcp[:3, 3] = [.02, -.01, .08]
    tcp[:3, :3] = Rotation.from_rotvec([.1, -.2, .05]).as_matrix()
    ik = PiperIK(tcp=tcp)
    nominal = PiperIK()
    np.testing.assert_allclose(ik.fk(Q), nominal.fk(Q) @ tcp)
    target = ik.fk(Q)
    target[0, 3] += .003
    result = ik.solve(target, Q)
    assert result.converged
    assert result.position_error < 1e-4
    assert np.linalg.norm(ik.fk(result.q)[:3, 3]-target[:3, 3]) < 1e-4


def test_custom_backend_receives_tool0_and_cannot_mutate_caller():
    captured = []
    class Backend:
        def solve(self, target, seed):
            captured.append(target.copy())
            target[:] = 0
            return seed
    tcp = np.eye(4); tcp[2, 3] = .1
    ik = PiperIK(Backend(), tcp=tcp)
    target = ik.fk(Q); target_before = target.copy(); q = Q.copy()
    result = ik.solve(target, q)
    np.testing.assert_allclose(captured[0], PiperIK().fk(Q))
    np.testing.assert_array_equal(target, target_before)
    np.testing.assert_array_equal(q, Q)
    assert result.converged
    target[0, 3] += 1
    assert not ik.solve(target, q).converged


def test_invalid_backend_result_is_rejected():
    class Broken:
        def solve(self, target, seed):
            return seed * np.nan
    ik = PiperIK(Broken())
    with pytest.raises(ValueError): ik.solve(ik.fk(Q), Q)


@pytest.mark.parametrize('bad', [np.ones((3, 3)), np.eye(4)*2, np.full((4, 4), np.nan), np.diag([-1, 1, 1, 1])])
def test_invalid_poses(bad):
    ik = PiperIK()
    with pytest.raises(ValueError): ik.solve(bad, Q)
    with pytest.raises(ValueError): ThirdOrderPoseFilter(bad)


@pytest.mark.parametrize('bad', [[0]*5, [0]*5+[np.nan], [100]*6])
def test_invalid_joints(bad):
    with pytest.raises(ValueError): PiperIK().fk(bad)


@pytest.mark.parametrize('kwargs', [{'max_iterations':0}, {'damping':0}, {'weights':(1,2)}, {'seed_weight':-1}, {'position_tolerance':float('nan')}])
def test_invalid_pin_config(kwargs):
    with pytest.raises(ValueError): PinocchioConfig(**kwargs)


def test_invalid_rl_config_and_model():
    with pytest.raises(ValueError): RLConfig(time_budget_s=0)
    with pytest.raises(ValueError): PiperIK('rl', urdf=ROOT/'piper_ik/assets/piper.urdf')


def test_filter_step_is_three_cascaded_stages_not_one():
    initial = np.eye(4); target = initial.copy(); target[0, 3] = 1
    f = ThirdOrderPoseFilter(initial, tau=.06)
    result = f.update(target, .02)
    alpha = 1-np.exp(-.02/.06)
    assert result[0, 3] == pytest.approx(alpha**3)
    result[:] = 0
    assert f.value[3, 3] == 1
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
        assert 0 <= angle <= np.deg2rad(2)+1e-12
        np.testing.assert_allclose(out[:3, :3].T @ out[:3, :3], np.eye(3), atol=1e-12)
    f.reset(b); np.testing.assert_allclose(f.update(b, .01), b, atol=1e-12)


@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf])
def test_filter_invalid_time_does_not_change_state(value):
    f = ThirdOrderPoseFilter(np.eye(4))
    with pytest.raises(ValueError): f.update(np.eye(4), value)
    np.testing.assert_array_equal(f.value, np.eye(4))
    with pytest.raises(ValueError): ThirdOrderPoseFilter(np.eye(4), tau=value)


def test_filter_speed_caps():
    target = np.eye(4); target[0, 3] = 1
    target[:3, :3] = Rotation.from_euler('z', 1).as_matrix()
    f = ThirdOrderPoseFilter(np.eye(4), tau=.001, linear_speed=.02, angular_speed=.1)
    for _ in range(10):
        before = f.value; after = f.update(target, .02)
        assert np.linalg.norm(after[:3,3]-before[:3,3]) <= .02*.02+1e-12
        assert Rotation.from_matrix(after[:3,:3]@before[:3,:3].T).magnitude() <= .1*.02+1e-12


def test_filter_and_ik_composition():
    ik = PiperIK(); q = Q.copy(); goal = ik.fk(q); goal[0,3] += .005
    f = ThirdOrderPoseFilter(ik.fk(q))
    for _ in range(60):
        result = ik.solve(f.update(goal, .02), q)
        assert result.converged
        q = result.q
    assert np.linalg.norm(ik.fk(q)[:3,3]-goal[:3,3]) < 1e-4


def test_training_and_runtime_actor_agree():
    torch = pytest.importorskip('torch')
    spec = importlib.util.spec_from_file_location('training_actor', ROOT/'training/train_rl.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from piper_ik._policy import Policy
    checkpoint = torch.load(ROOT/'piper_ik/assets/ppo_precision.pt', map_location='cpu', weights_only=False)
    actor = module.ActorCritic(centered=True); actor.load_state_dict(checkpoint['model'])
    ik = PiperIK(); lo, hi = ik.joint_limits; policy = Policy(ROOT/'piper_ik/assets/ppo_precision.pt', lo, hi)
    T = ik.fk(Q); G = T.copy(); G[:3,3] += [.004,-.006,.002]
    G[:3,:3] = Rotation.from_rotvec([.01,-.03,.02]).as_matrix() @ T[:3,:3]
    def quat(R):
        q = np.roll(Rotation.from_matrix(R).as_quat(),1)
        return q if q[0]>=0 else -q
    obs = np.r_[T[:3,3],quat(T[:3,:3]),G[:3,3],quat(G[:3,:3]),Q,
                (G[:3,3]-T[:3,3])*20,Rotation.from_matrix(G[:3,:3]@T[:3,:3].T).as_rotvec()*3].astype('float32')
    with torch.no_grad(): delta = actor.mean(torch.from_numpy(obs)[None])[0].clamp(-1,1).numpy()*.2
    np.testing.assert_allclose(policy.propose(Q,T,G), np.clip(Q+delta,lo,hi), atol=2e-7)


def test_packaged_mujoco_training_model_matches_fk():
    mj = pytest.importorskip('mujoco')
    model = mj.MjModel.from_xml_path(str(ROOT/'piper_ik/assets/piper.xml')); data = mj.MjData(model)
    ik = PiperIK(); lo,hi = ik.joint_limits; rng = np.random.default_rng(14)
    for q in rng.uniform(lo,hi,(20,6)):
        data.qpos[:]=q; mj.mj_kinematics(model,data); body=model.body('tool0').id; T=ik.fk(q)
        np.testing.assert_allclose(T[:3,3],data.xpos[body],atol=2e-6)
        np.testing.assert_allclose(T[:3,:3],data.xmat[body].reshape(3,3),atol=2e-5)


def test_feedback_fk_preserves_out_of_limit_measurement_without_relaxing_ik():
    ik = PiperIK()
    lo, hi = ik.joint_limits
    q = (lo + hi) / 2
    q[4] = hi[4] + 3e-6
    with pytest.raises(ValueError):
        ik.fk(q)
    measured = ik.fk(q, check_limits=False)
    assert np.isfinite(measured).all()
    assert not np.array_equal(measured, ik.fk(np.clip(q, lo, hi)))
    with pytest.raises(ValueError):
        ik.solve(measured, q)
    q[0] = np.nan
    with pytest.raises(ValueError):
        ik.fk(q, check_limits=False)
