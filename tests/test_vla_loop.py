"""examples/vla_loop.py exactly as Codex copies it, on RealtimeArm('sim'): the control process at 200 Hz against the
simulated arm on its CAN socket, and a policy that moves forward from whatever state it observes (3 s per case)."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from piper_ik import RealtimeArm

spec = importlib.util.spec_from_file_location('vla_loop', Path(__file__).resolve().parents[1] / 'examples' / 'vla_loop.py')
vla_loop = importlib.util.module_from_spec(spec); spec.loader.exec_module(vla_loop)


class Forward:
    """Closed loop: 16 actions, 1.5 mm per frame forward from the observed state, gripper closing past 3 cm."""
    def __init__(self): self.calls = 0
    def infer(self, images, state):
        self.calls += 1; a = np.tile(state, (16, 1)); a[:, 0] = state[0] + .0015 * np.arange(1, 17); a[:, 1:6] = 0.
        a[:, 6] = np.where(a[:, 0] > .03, 0., 1.); return a


def test_template_runs_at_200_hz_in_its_own_process():
    policy = Forward()
    with RealtimeArm('sim', keep_stats=True) as arm:
        vla_loop.run(arm, policy, lambda: None, seconds=3.)
        state, t = arm.observation(); late, cycles, ik = arm.late_cycles, arm.cycles, arm.last_ik; sent = dict(arm.sim.commands)
    itv = arm.stats['interval_s']; hz = len(itv) / itv.sum()
    print(f'{cycles} cycles at {hz:.1f} Hz ({late} late), {policy.calls} chunks, data time {t:.2f} s, moved {state[0] * 100:.1f} cm, gripper {state[6]:.2f}')
    assert abs(hz - 200.) < 2. and cycles > 550 and policy.calls >= 5 and t > 2.        # a chunk every 10 frames = 0.5 s
    n = sent[0x151]; assert abs(n - cycles) <= 3                                            # one cycle = mode + 3 joint frames + gripper
    assert all(abs(sent[k] - n) <= 1 for k in (0x155, 0x156, 0x157)) and abs(sent[0x159] - (n + 2)) <= 1   # + clear errors / enable
    assert state[0] > .05 and abs(state[1]) < 1e-3 and state[6] < .01                  # moved forward on the policy's line and closed
    assert ik.position_error < 1e-4


def test_policy_errors_end_the_loop_and_the_arm_holds():
    class Broken:
        def infer(self, images, state): raise ValueError('model failed')
    with RealtimeArm('sim') as arm:
        with pytest.raises(ValueError, match='model failed'):
            vla_loop.run(arm, Broken(), lambda: None, seconds=3.)
        q, _ = arm.command; assert np.isfinite(q).all() and arm.cycles > 0               # control process still holding the start pose


def test_control_process_errors_surface():
    with pytest.raises(RuntimeError, match='failed to start'):
        RealtimeArm('unix:\0piper_ik_no_such_bus', start_timeout=20.).start()
