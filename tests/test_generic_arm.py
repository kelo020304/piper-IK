"""Another arm from its URDF: tests/data/arm7.urdf (7 revolute joints, other names, a finger off the chain, TCP frame
"tcp"). Model detection, IK, the FastUMI chain, RealtimeArm with SimDriver, the check tool, and the Piper preset."""
import numpy as np
import pytest
from pathlib import Path
from piper_ik import ArmIK, PiperIK, RealtimeArm, SimDriver, MODEL, fastumi_to_robot, robot_to_fastumi, tcp_axes_matrix
from piper_ik import check

ARM7 = Path(__file__).resolve().parent / 'data' / 'arm7.urdf'
READY = np.array([0., -.785, 0., -2.356, 0., 1.571, .785])        # TCP 0.49 m up, pointing down
LEVEL = np.array([.7788, -.7333, -.7797, -2.6146, -2.8973, 2.5854, .0824])   # TCP at (0.5, 0, 0.45) m, level, pointing along base +x
# arm7's TCP has the Piper's convention (z = approach, y = finger axis), so the default tcp_axes fit it


def arm7(**kw):
    return ArmIK(ARM7, tcp_frame='tcp', **kw)


def test_model_from_the_urdf():
    ik = arm7(); lo, hi = ik.joint_limits
    assert ik.dof == 7 and ik.joint_names == [f'a{i}' for i in range(1, 8)]          # the finger is locked
    assert ik.ik_frame == 'l7' and np.allclose(ik.tcp[:3, 3], [0, 0, .107 + .1034], atol=1e-12)
    assert np.allclose(lo[3], -3.0718) and np.allclose(hi[5], 3.7525)
    T = ik.fk(READY); assert np.allclose(T[:3, 2], [0, 0, -1], atol=1e-3) and abs(T[2, 3] - .487) < 1e-3


def test_ik_round_trip_inside_the_limits():
    ik = arm7(); lo, hi = ik.joint_limits; rng = np.random.default_rng(0); ok = 0
    for _ in range(100):
        q = rng.uniform(lo + .1, hi - .1); r = ik.solve(ik.fk(q), np.clip(q + rng.normal(size=7) * .05, lo, hi))
        ok += r.converged; assert np.all(r.q >= lo) and np.all(r.q <= hi) and r.position_error < 2e-3
    assert ok >= 95


def test_explicit_frames_joints_and_tcp():
    """ik_frame / joints / tcp given by hand give the same model as found from the URDF."""
    auto = arm7(); T = np.eye(4); T[:3, :3] = auto.tcp[:3, :3]; T[:3, 3] = auto.tcp[:3, 3]
    hand = ArmIK(ARM7, tcp_frame='l7', ik_frame='l7', joints=[f'a{i}' for i in range(1, 8)], tcp=T)
    for q in np.random.default_rng(1).uniform(*auto.joint_limits, (20, 7)):
        assert np.allclose(hand.fk(q), auto.fk(q), atol=1e-12)
    fewer = ArmIK(ARM7, tcp_frame='tcp', joints=[f'a{i}' for i in range(1, 7)])      # a7 locked at 0
    assert fewer.dof == 6 and np.allclose(fewer.fk(READY[:6]), auto.fk(np.r_[READY[:6], 0.]), atol=1e-12)


def test_bad_models_are_refused(tmp_path):
    with pytest.raises(ValueError, match='tcp_frame'):
        ArmIK(ARM7, tcp_frame='tool0')
    with pytest.raises(ValueError, match='not in the URDF'):
        ArmIK(ARM7, tcp_frame='tcp', joints=['a1', 'nope'])
    cont = tmp_path / 'cont.urdf'; cont.write_text(ARM7.read_text().replace('<joint name="a7" type="revolute">', '<joint name="a7" type="continuous">'))
    with pytest.raises(ValueError, match='revolute or prismatic'):
        ArmIK(cont, tcp_frame='tcp')
    with pytest.raises(ValueError, match='7-vectors'):
        arm7(limits=(np.zeros(6), np.ones(6)))


def test_tcp_axes_words_and_matrix():
    assert np.array_equal(tcp_axes_matrix(('down', 'left', 'forward')), tcp_axes_matrix(None))
    A = tcp_axes_matrix(('forward', 'right', 'down')); assert np.array_equal(tcp_axes_matrix(A), A)
    with pytest.raises(ValueError, match='right-handed'):
        tcp_axes_matrix(('forward', 'left', 'down'))
    with pytest.raises(ValueError, match='three of'):
        tcp_axes_matrix(('ahead', 'left', 'up'))


def test_fastumi_axes_follow_tcp_axes():
    """FastUMI x (forward) moves the TCP along whichever TCP axis tcp_axes calls forward; the inverse round-trips."""
    rng = np.random.default_rng(2); anchor = arm7().fk(READY)
    for words in (('down', 'left', 'forward'), ('forward', 'right', 'down'), ('left', 'up', 'forward'), ('up', 'right', 'forward')):
        A = tcp_axes_matrix(words); fwd = int(np.argmax(np.abs(A[0]))); sign = np.sign(A[0, fwd])
        T = fastumi_to_robot(np.array([.05, 0, 0, 0, 0, 0]), anchor, axes=A)[0]
        assert np.allclose(T[:3, 3] - anchor[:3, 3], .05 * sign * anchor[:3, fwd], atol=1e-12)
        row = np.r_[rng.uniform(-.1, .1, 3), rng.uniform(-.5, .5, 3)]
        assert np.allclose(robot_to_fastumi(fastumi_to_robot(row, anchor, axes=A)[0], anchor, axes=A), row, atol=1e-12)


def test_chain_on_another_arm():
    """The FastUMI chain (chunks, strict timing) on the 7-joint arm."""
    ik = arm7(tcp_axes=('down', 'left', 'forward'), follow='strict'); ik.start(READY, .07)
    a = np.zeros((10, 7)); a[:, 0] = .004 * np.arange(1, 11); a[:, 6] = 1.        # 4 cm forward along FastUMI x
    ik.push_actions(a, 0.); q = READY.copy()
    for _ in range(100):
        q, w = ik.tick(q, .005)
    assert q.shape == (7,) and abs(w - .07) < 1e-12 and ik.finished
    state, _ = ik.observation(q, .07)
    assert np.allclose(state[:3], [.04, 0, 0], atol=2e-4)
    assert np.allclose(ik.fk(q)[:3, 3] - ik.fk(READY)[:3, 3], [0, 0, -.04], atol=2e-4)   # ready pose: TCP z down, so forward = down


def test_realtime_arm_with_another_arm_and_sim_driver():
    robot = dict(urdf=str(ARM7), tcp_frame='tcp')
    with RealtimeArm(robot=robot, driver=SimDriver(READY), keep_stats=True) as arm:
        state, t = arm.observation(); a = np.tile(state, (16, 1)); a[:, 0] = .002 * np.arange(1, 17)
        arm.push_actions(a, t); arm.wait_until(t + 16 / arm.fps + .5)
        q_cmd, width = arm.command; joints = arm.joints; state, _ = arm.observation(); cycles = arm.cycles
    itv = arm.stats['interval_s']; hz = len(itv) / itv.sum()
    assert q_cmd.shape == (7,) and joints.shape == (7,) and abs(width - .07) < 1e-12
    assert abs(hz - 200.) < 3. and cycles > 100 and abs(state[0] - .032) < 1e-3 and abs(state[1]) < 1e-3
    assert arm.fk(joints)[2, 3] < arm.fk(READY)[2, 3] - .03                        # forward = down at the ready pose


def test_another_arm_needs_its_driver():
    with pytest.raises(ValueError, match='needs its own driver'):
        RealtimeArm(robot=dict(urdf=str(ARM7), tcp_frame='tcp'))


def test_check_tool(capsys):
    assert check.main([str(ARM7), '--tcp-frame', 'tcp', '--n', '40', '--q0', ','.join(map(str, LEVEL))]) == 0
    out = capsys.readouterr().out
    assert "locked      ['finger_joint']" in out and 'IK frame    l7' in out and 'tcp_axes = down,left,forward if' in out


def test_piper_preset_is_armik_on_the_bundled_model():
    """ArmIK on the bundled URDF finds link6 and joint1..6 by itself and solves exactly as PiperIK."""
    a, p = ArmIK(MODEL), PiperIK(); assert a.ik_frame == p.ik_frame == 'link6' and a.joint_names == p.joint_names
    assert np.array_equal(a.tcp, p.tcp)
    rng = np.random.default_rng(3); lo, hi = p.joint_limits
    for _ in range(20):
        q = rng.uniform(lo, hi); T = p.fk(q); s = np.clip(q + rng.normal(size=6) * .3, lo, hi)
        assert np.array_equal(a.solve(T, s).q, p.solve(T, s).q)
