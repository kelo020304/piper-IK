"""Standalone checks (numpy / scipy only): model, FK / IK, FastUMI mapping, interpolation without smoothing,
chunk timing, 追点 pacing, gripper mapping."""
import sys
import threading
import numpy as np
from scipy.spatial.transform import Rotation
import piper_ik
from piper_ik import PiperIK, hardware_limits, fastumi_to_robot, robot_to_fastumi, gripper_width, gripper_value
from piper_ik.trajectory import Trajectory

Q0 = np.array([0., 1.0, -1.2, 0., .6, 0.])         # a working posture, gripper ahead of the base


def test_pinocchio_is_the_pip_wheel_even_with_ros_on_the_path():
    """The import works in a shell with ROS noetic sourced (its eigenpy would otherwise break Pinocchio)."""
    import pinocchio
    assert pinocchio.__version__ == '3.7.0' and 'cmeel.prefix' in pinocchio.__file__ and piper_ik.MODEL.is_file()


def test_model_limits_and_tcp():
    ik = PiperIK(); lo, hi = ik.joint_limits; elo, ehi = hardware_limits()
    assert np.allclose(lo, elo, atol=1e-12) and np.allclose(hi, ehi, atol=1e-12)
    assert np.allclose(np.degrees(hi), [150, 179.9, 0, 100, 70, 120]) and np.allclose(np.degrees(lo), [-150, 0, -170, -100, -70, -120])
    assert abs(ik.tcp[2, 3] - .2171) < 1e-4                                       # UMI fingertip 217 mm ahead of the flange
    assert ik.joint_names == [f'joint{i}' for i in range(1, 7)]


def test_ik_round_trip():
    """The solver stops when its line search makes no progress (as the console's does): a few postures end just
    above the 0.1 mm tolerance."""
    ik = PiperIK(); lo, hi = ik.joint_limits; rng = np.random.default_rng(0); res = []
    for _ in range(100):
        q = rng.uniform(lo + .1, hi - .1); seed = np.clip(q + rng.normal(size=6) * .05, lo, hi)
        r = ik.solve(ik.fk(q), seed); res.append((r.converged, r.position_error))
        assert np.all(r.q >= lo) and np.all(r.q <= hi)
    assert sum(c for c, _ in res) >= 95 and max(e for _, e in res) < 1e-3


def test_unreachable_target_returns_best_effort_inside_limits():
    ik = PiperIK(); lo, hi = ik.joint_limits; T = ik.fk(Q0); T[:3, 3] += [1., 0, 0]   # 1 m further than the arm reaches
    r = ik.solve(T, Q0)
    assert not r.converged and r.position_error > .3 and np.all(r.q >= lo) and np.all(r.q <= hi)


def test_firmware_limits_narrow_the_bounds():
    fw = [None] * 6; fw[4] = (-np.deg2rad(60), np.deg2rad(60))
    ik = PiperIK(hardware_limits(fw)); assert np.isclose(np.degrees(ik.joint_limits[1][4]), 60)


def test_mapping_round_trip():
    rng = np.random.default_rng(0); anchor = np.eye(4); anchor[:3, 3] = [.4, .05, .1]
    for _ in range(50):
        row = np.r_[rng.uniform(-.2, .2, 3), rng.uniform(-1, 1, 3)]
        back = robot_to_fastumi(fastumi_to_robot(row, anchor)[0], anchor)
        assert np.allclose(back[:3], row[:3], atol=1e-12)
        assert np.allclose(Rotation.from_euler('xyz', back[3:]).as_matrix(), Rotation.from_euler('xyz', row[3:]).as_matrix(), atol=1e-12)


def test_fastumi_axes_on_a_level_gripper():
    """A level gripper (tool0 z = base +x, x = base -z): FastUMI x forward / y left / z up = base +x / +y / +z."""
    anchor = np.eye(4); anchor[:3, :3] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]); anchor[:3, 3] = [.4, 0., .1]
    for axis in range(3):
        row = np.zeros(6); row[axis] = .05
        assert np.allclose(fastumi_to_robot(row, anchor)[0][:3, 3] - anchor[:3, 3], np.eye(3)[axis] * .05, atol=1e-12)


def test_points_are_passed_through_exactly():
    """At a point's own time the target is that point: no filter, corners stay corners."""
    rng = np.random.default_rng(1); poses = []
    for _ in range(5):
        T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(rng.uniform(-.5, .5, 3)).as_matrix(); T[:3, 3] = rng.uniform(-.1, .1, 3); poses.append(T)
    tr = Trajectory([0., .05, .1, .15, .2], poses, [1., .8, .2, 0., 0.])
    for t, T, g in zip(tr.times, tr.poses, tr.grip):
        P, G = tr.at(t)
        assert np.allclose(P, T, atol=1e-12) and abs(G - g) < 1e-12
    P, _ = tr.at(.025)
    assert np.allclose(P[:3, 3], (poses[0][:3, 3] + poses[1][:3, 3]) / 2, atol=1e-12)


def chunk(n=10, step_m=.004):
    a = np.zeros((n, 7)); a[:, 0] = step_m * np.arange(1, n + 1); a[:, 6] = 1.     # move forward, gripper open
    return a


def test_chunk_timing_and_strict_tracking():
    ik = PiperIK(follow='strict'); ik.start(Q0, .07); ik.push_actions(chunk(), 0.)
    assert abs(ik.end_time - 10 / 20) < 1e-12                                     # action k at t_obs + (k+1)/20 s
    q = Q0.copy()
    for _ in range(25):
        q, w = ik.tick(q, .01)                                                    # ideal arm: measured = command
    assert abs(ik.source_time - .25) < 1e-9 and abs(w - .07) < 1e-12
    assert np.allclose(ik.fk(q)[:3, 3], ik.to_robot(chunk()[4, :6])[0][:3, 3], atol=2e-4)   # 0.25 s = action 4


def test_new_chunk_replaces_the_rest_of_the_old_one():
    ik = PiperIK(follow='strict'); ik.start(Q0, .07); ik.push_actions(chunk(), 0.); q = Q0.copy()
    for _ in range(10):
        q, _ = ik.tick(q, .01)
    state, t_obs = ik.observation(q, .07)
    assert abs(t_obs - .1) < 1e-9 and abs(state[0] - .008) < 2e-4 and abs(state[6] - 1.) < 1e-12
    b = chunk(); b[:, 1] = .002 * np.arange(1, 11)
    ik.push_actions(b, t_obs)
    assert abs(ik.end_time - (t_obs + .5)) < 1e-12
    for _ in range(5):
        q, _ = ik.tick(q, .01)
    assert np.allclose(ik.fk(q)[:3, 3], ik.to_robot(b[0, :6])[0][:3, 3], atol=2e-4)    # t_obs + 0.05 s = new action 0


def test_hold_at_the_end_until_the_next_chunk():
    ik = PiperIK(follow='strict'); ik.start(Q0, .07); ik.push_actions(chunk(3), 0.); q = Q0.copy()
    for _ in range(40):
        q, _ = ik.tick(q, .01)
    assert ik.finished and abs(ik.source_time - .15) < 1e-12


def test_wait_mode_stops_data_time_when_the_arm_lags():
    """追点: an arm that does not move -> lead grows -> data time crawls at 0.1x; the targets are unchanged."""
    ik = PiperIK(follow='wait'); ik.start(Q0, .07); ik.push_actions(chunk(20, .01), 0.)
    for _ in range(50):
        ik.tick(Q0, .01)
    assert ik.follow_scale == .1 and ik.source_time < .2


def test_threads_tick_and_push():
    ik = PiperIK(follow='strict'); ik.start(Q0, .07); ik.push_actions(chunk(), 0.); stop = threading.Event(); errors = []
    def pusher():
        try:
            while not stop.is_set():
                state, t = ik.observation(Q0, .07); ik.push_actions(chunk(), t)
        except Exception as e:                     # surfaced by the assert below
            errors.append(e)
    th = threading.Thread(target=pusher); th.start(); q = Q0.copy()
    for _ in range(200):
        q, _ = ik.tick(q, .01)
    stop.set(); th.join()
    assert not errors and np.isfinite(q).all()


def _long_session(monkeypatch, history_s, seconds=70.):
    """A policy run with 23-25 s grasps (gripper value jittering by 0.03): chunks of 16 every 4 frames, an arm that
    chases its target acceleration-limited (examples/dry_run.py). Returns every command and the final history length."""
    monkeypatch.setattr(piper_ik.core, 'PUSH_HISTORY_S', history_s)
    def row(t):
        g = 0. if 5 <= t < 30 or 45 <= t < 68 else 1.
        return np.array([.04 * np.sin(2 * np.pi * t / 14), .03 * np.sin(2 * np.pi * t / 9), -.01 * (1 - np.cos(2 * np.pi * t / 20)),
                         0., np.deg2rad(8) * np.sin(2 * np.pi * t / 11), 0., .03 + .94 * g + .02 * np.sin(13.7 * t) + .01 * np.sin(41.3 * t)])
    ik = PiperIK(); ik.start(Q0, .07); q, v, goal = Q0.copy(), np.zeros(6), Q0.copy(); nxt = 0.; out = []
    while ik.source_time < seconds:
        if ik.source_time >= nxt:
            _, t_obs = ik.observation(q, .07); ik.push_actions(np.array([row(t_obs + (k + 1) / 20) for k in range(16)]), t_obs); nxt = t_obs + .2
        goal, _ = ik.tick(q, .01); out.append(goal)
        err = goal - q; want = np.clip(np.sign(err) * np.sqrt(10 * np.abs(err)), -3, 3); v += np.clip(want - v, -.05, .05)
        arrive = np.abs(v * .01) >= np.abs(err); q = np.where(arrive, goal, q + v * .01); v = np.where(arrive, 0., v)
    return np.array(out), len(ik._traj.times)


def test_long_sessions_keep_a_bounded_history_with_the_same_commands(monkeypatch):
    """push_actions drops points 30 s before t_obs: each chunk costs the same however long the policy runs, and
    the commands (targets, 追点 pacing, grasp / release moments) are bit-identical to keeping everything."""
    full, n_full = _long_session(monkeypatch, np.inf)
    kept, n_kept = _long_session(monkeypatch, 30.)
    assert np.array_equal(full, kept) and n_full > 1500 and n_kept < 800


def test_gripper_mapping():
    assert gripper_width(0.) == 0. and abs(gripper_width(1.) - .07) < 1e-15 and abs(gripper_width(.5) - .035) < 1e-15
    assert gripper_width(1.2) == gripper_width(1.) and gripper_width(-.1) == 0.
    assert abs(gripper_value(.035) - .5) < 1e-12
