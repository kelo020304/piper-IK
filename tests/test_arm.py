"""PiperArm against a fake piper_sdk: the same calls, order and units as the data_viewer console (device.py)."""
import sys
import time
import types
from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_ik import PiperArm, hardware_limits


class FakeSDK:
    """Records calls; joints and firmware limits as the controller reports them (0.001 deg / 0.1 deg)."""
    def __init__(self, can, **kw):
        self.can, self.kw, self.calls = can, kw, []; self.q_mdeg = [1000, 60000, -70000, 0, 30000, -5000]; self.enabled = False
        self.motor = 1; self.limit_stamp = 0.; self.angle = {5: (-650, 650)}          # J5 firmware limit +-65 deg
    def ConnectPort(self, **kw): self.calls.append(('connect', kw))
    def DisconnectPort(self): self.calls.append(('disconnect',))
    def GetArmJointMsgs(self): return NS(time_stamp=time.monotonic(), joint_state=NS(**{f'joint_{i + 1}': v for i, v in enumerate(self.q_mdeg)}))
    def MotionCtrl_1(self, *a): self.calls.append(('stop', a))
    def EnableArm(self, n): self.calls.append(('enable', n)); self.enabled = True
    def DisableArm(self, n): self.calls.append(('disable', n)); self.enabled = False
    def GetArmEnableStatus(self): return [self.enabled] * 6
    def SearchMotorMaxAngleSpdAccLimit(self, j, kind): self.calls.append(('query_limit', j, kind)); self.motor = j; self.limit_stamp = time.monotonic()
    def GetCurrentMotorAngleLimitMaxVel(self):
        lo, hi = self.angle.get(self.motor, (0x7FFF, 0x7FFF))
        return NS(time_stamp=self.limit_stamp, current_motor_angle_limit_max_vel=NS(motor_num=self.motor, max_joint_spd=3000, min_angle_limit=lo, max_angle_limit=hi))
    def MotionCtrl_2(self, *a): self.calls.append(('mode', a))
    def JointCtrl(self, *a): self.calls.append(('joints', a))
    def GetArmGripperMsgs(self): return NS(time_stamp=time.monotonic(), gripper_state=NS(grippers_angle=12000, grippers_effort=0))
    def GripperCtrl(self, *a): self.calls.append(('gripper', a))


@pytest.fixture
def arm(monkeypatch):
    made = []
    def factory(can, **kw):
        made.append(FakeSDK(can, **kw)); return made[-1]
    monkeypatch.setitem(sys.modules, 'piper_sdk', types.SimpleNamespace(C_PiperInterface_V2=factory))
    a = PiperArm('can7'); a.connect()
    return a, made[0]


def test_connect_without_the_sdk_joint_clamp(arm):
    a, sdk = arm
    assert sdk.can == 'can7' and sdk.kw == {'start_sdk_joint_limit': False} and sdk.calls == [('connect', {'piper_init': False})]


def test_enable_reads_firmware_limits_and_narrows_them(arm):
    a, sdk = arm; lo, hi = a.enable(speed_percent=100)
    assert all(c[0] != 'stop' for c in sdk.calls) and ('enable', 7) in sdk.calls                 # never 'resume'
    assert [c[1] for c in sdk.calls if c[0] == 'query_limit'] == [1, 2, 3, 4, 5, 6]
    elo, ehi = hardware_limits()
    assert np.isclose(np.degrees(hi[4]), 65) and np.isclose(np.degrees(lo[4]), -65)           # J5 narrowed by the firmware
    assert np.allclose(np.delete(lo, 4), np.delete(elo, 4)) and np.allclose(np.delete(hi, 4), np.delete(ehi, 4))


def test_an_enabled_arm_gets_no_enable_or_resume(arm):
    """2026-10-16: resume + enable sent to an already enabled arm dropped every motor. Now: nothing is sent."""
    a, sdk = arm; sdk.enabled = True; a.enable(speed_percent=100)
    assert [c for c in sdk.calls if c[0] in ('stop', 'enable', 'disable')] == []


def test_joints_and_gripper_units(arm):
    a, sdk = arm
    assert np.allclose(np.degrees(a.joints()), [1, 60, -70, 0, 30, -5]) and abs(a.gripper_width() - .012) < 1e-12


def test_send_joints_is_move_j_at_100_percent_in_millidegrees(arm):
    a, sdk = arm; a.enable(); n = len(sdk.calls)
    a.send_joints(np.deg2rad([1.0004, 60, -70.0006, 0, 30, -5]))
    assert sdk.calls[n:] == [('mode', (0x01, 0x01, 100, 0x00)), ('joints', (1000, 60000, -70001, 0, 30000, -5000))]


def test_gripper_prepared_once_then_commanded_in_micrometres(arm):
    a, sdk = arm; a.enable(); n = len(sdk.calls)
    a.send_gripper(.07); a.send_gripper(.035)
    assert sdk.calls[n:] == [('gripper', (12000, 1000, 0x02, 0x00)), ('gripper', (12000, 1000, 0x01, 0x00)),
                             ('gripper', (70000, 1000, 0x01, 0x00)), ('gripper', (35000, 1000, 0x01, 0x00))]


def test_command_frames_are_the_bytes_piper_sdk_sends():
    """RealtimeArm packs each cycle itself: the same five frames, byte for byte, as PiperArm's SDK calls."""
    sdk_mod = pytest.importorskip('piper_sdk')
    from piper_sdk.hardware_port import C_STD_CAN
    from piper_ik.arm import command_frames, gripper_frame, joint_units
    sent = []

    class Recorder(C_STD_CAN):
        def __init__(self): super().__init__('rec', 'socketcan', 1000000, False, False, None)
        def SendCanMessage(self, arbitration_id, data, dlc=8, is_extended_id=False):
            sent.append((arbitration_id, bytes(data))); return self.CAN_STATUS.SEND_MESSAGE_SUCCESS
    sdk = sdk_mod.C_PiperInterface_V2('piper_ik_frame_test', judge_flag=False, can_auto_init=False, start_sdk_joint_limit=False)
    sdk._C_PiperInterface_V2__arm_can = Recorder()
    rng = np.random.default_rng(1); lo, hi = hardware_limits()
    for _ in range(200):
        q = rng.uniform(lo, hi); w = rng.uniform(0, .07); speed = int(rng.integers(1, 101))
        sent.clear(); sdk.MotionCtrl_2(0x01, 0x01, speed, 0x00); sdk.JointCtrl(*joint_units(q)); sdk.GripperCtrl(int(round(w * 1e6)), 1000, 0x01, 0x00)
        assert sent == command_frames(q, w, speed, 1000)
        sent.clear(); sdk.GripperCtrl(int(round(w * 1e6)), 1000, 0x02, 0x00); assert sent == [gripper_frame(w, 1000, 0x02)]


def test_raw_feedback_reads_as_piper_arm_does():
    import socket
    import struct
    from piper_ik import PiperCAN
    from piper_ik.arm import CAN_FRAME
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET); io = PiperCAN(a)
    mdeg = [1000, 60000, -70000, 0, 30000, -5000]
    for j in range(3):
        b.send(CAN_FRAME.pack(0x2A5 + j, 8, struct.pack('>ii', *mdeg[2 * j:2 * j + 2])))
    assert not io.ready
    b.send(CAN_FRAME.pack(0x2A8, 8, struct.pack('>ihBB', 12000, 0, 0, 0))); b.send(CAN_FRAME.pack(0x2A1, 8, bytes(8)))
    io.poll()
    assert io.ready and io.received == 5
    assert np.array_equal(io.joints(), np.deg2rad(np.array(mdeg, float) / 1000)) and io.gripper_width() == 12000 / 1e6   # PiperArm's arithmetic
    io.send([(0x155, struct.pack('>ii', 1, -2))]); cid, n, data = CAN_FRAME.unpack(b.recv(16))
    assert (cid, n, struct.unpack('>ii', data)) == (0x155, 8, (1, -2))
