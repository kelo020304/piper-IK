"""Thin piper_sdk bridge making the same calls as the data_viewer console (device.py).

  connect   C_PiperInterface_V2(can, start_sdk_joint_limit=False); ConnectPort(piper_init=False)
            (the SDK-side joint clamp is off: it clips commands and feedback to a fixed table)
  enable    after fresh status frames: already enabled -> send nothing; disabled -> EnableArm(7) only, until three fresh
            reads show all six; emergency stop -> refuse. Never MotionCtrl_1(0x02) ('resume'): sent to an enabled arm
            it dropped every motor (2026-10-16). Then read the firmware angle limits (queries only)
  send      MotionCtrl_2(0x01, 0x01, speed_percent, 0x00) + JointCtrl(0.001 deg) every cycle (MOVE J)
  gripper   GripperCtrl(width 0.001 mm, effort, 0x01, 0x00), after the official 0x02 -> 0x01 preparation
Speed percent 100: the controller's own speed scaling is the only remaining slow-down layer; keep it at 100.

PiperCAN is the per-cycle part of the same traffic without piper_sdk (used by RealtimeArm): one raw SocketCAN socket
that the kernel filters to the joint and gripper feedback, read without blocking each cycle, and the command frames
packed byte for byte as piper_sdk packs them.
"""
import socket
import struct
import time
import numpy as np
from .core import hardware_limits
from .convention import GRIPPER_EFFORT_RAW

FEEDBACK_IDS = (0x2A5, 0x2A6, 0x2A7, 0x2A8)          # joints 1-2, 3-4, 5-6 (0.001 deg), gripper (0.001 mm): 200 Hz each
CAN_FRAME = struct.Struct('=IB3x8s')                  # struct can_frame: id, length, padding, 8 data bytes
STATUS_SETTLE_S = .6                                  # after connecting, wait this long before trusting the enable status


def joint_units(q):
    """Joint target (6,) rad -> the controller's 0.001 deg integers, as PiperArm.send_joints rounds them."""
    return [int(u) for u in np.rint(np.rad2deg(np.asarray(q, float)) * 1000).astype(int)]


def command_frames(q, width_m, speed_percent=100, effort_raw=GRIPPER_EFFORT_RAW, gripper_code=0x01):
    """One control cycle as (CAN id, 8 bytes): what MotionCtrl_2(0x01, 0x01, speed, 0x00) + JointCtrl(...) +
    GripperCtrl(width, effort, code, 0x00) send through piper_sdk (0x151, 0x155, 0x156, 0x157, 0x159)."""
    u = joint_units(q)
    return [(0x151, bytes((0x01, 0x01, int(speed_percent), 0x00, 0, 0, 0, 0))),
            (0x155, struct.pack('>ii', u[0], u[1])), (0x156, struct.pack('>ii', u[2], u[3])), (0x157, struct.pack('>ii', u[4], u[5])),
            gripper_frame(width_m, effort_raw, gripper_code)]


def gripper_frame(width_m, effort_raw=GRIPPER_EFFORT_RAW, code=0x01):
    return 0x159, struct.pack('>iHBB', int(round(width_m * 1e6)), int(effort_raw), code, 0x00)


class PiperCAN:
    """Per-cycle CAN I/O on a socket carrying struct can_frame records: a raw SocketCAN socket (socketcan()) or, for
    the simulated arm, a Unix SEQPACKET socket (unix()). poll() reads everything waiting without blocking; joints() /
    gripper_width() are the latest feedback in the units PiperArm returns; send() never blocks (a frame the kernel
    cannot queue is dropped and counted, as piper_sdk drops it)."""

    def __init__(self, sock):
        self.sock = sock; sock.setblocking(False); self._units = [None] * 6; self._width = None; self.dropped = 0; self.received = 0

    @classmethod
    def socketcan(cls, channel):
        s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        mask = socket.CAN_SFF_MASK | socket.CAN_EFF_FLAG | socket.CAN_RTR_FLAG          # standard data frames with exactly this id
        s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b''.join(struct.pack('=II', i, mask) for i in FEEDBACK_IDS))
        s.bind((channel,))
        return cls(s)

    @classmethod
    def unix(cls, address):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET); s.connect(address)
        return cls(s)

    def poll(self):
        while True:
            try:
                cf = self.sock.recv(CAN_FRAME.size)
            except BlockingIOError:
                return
            cid, _, data = CAN_FRAME.unpack(cf); cid &= socket.CAN_SFF_MASK; self.received += 1
            if 0x2A5 <= cid <= 0x2A7:
                j = 2 * (cid - 0x2A5); self._units[j], self._units[j + 1] = struct.unpack('>ii', data)
            elif cid == 0x2A8:
                self._width = struct.unpack('>i', data[:4])[0]

    @property
    def ready(self):
        """Every joint and the gripper reported at least once."""
        return None not in self._units and self._width is not None

    def joints(self):
        return np.deg2rad(np.array(self._units, float) / 1000)

    def gripper_width(self):
        return float(self._width) / 1e6

    def send(self, frames):
        for cid, data in frames:
            try:
                self.sock.send(CAN_FRAME.pack(cid, 8, data))
            except (BlockingIOError, OSError):
                self.dropped += 1

    def close(self):
        self.sock.close()


class PiperArm:
    def __init__(self, can='can0'):
        self.can = can; self.sdk = None; self.speed_percent = 100; self.gripper_ready = False; self.firmware_limits = None

    def connect(self):
        from piper_sdk import C_PiperInterface_V2
        self.sdk = C_PiperInterface_V2(self.can, start_sdk_joint_limit=False)
        self.sdk.ConnectPort(piper_init=False)
        deadline = time.monotonic() + 2.
        while float(self.sdk.GetArmJointMsgs().time_stamp) <= 0:          # wait for the first joint feedback
            if time.monotonic() > deadline:
                raise RuntimeError(f'no joint feedback on {self.can}: check power, CAN wiring and `ip link show {self.can}` (bitrate 1000000, UP)')
            time.sleep(.01)

    def enable(self, speed_percent=100):
        """Enable all joints if needed (no motion) and return the IK joint limits (lo, hi) in rad:
        datasheet ∩ SDK table ∩ this controller's firmware readback. Pass them to PiperIK(limits)."""
        if type(speed_percent) is not int or not 1 <= speed_percent <= 100:
            raise ValueError('speed_percent must be an int 1..100 (use 100)')
        self.speed_percent = speed_percent
        time.sleep(STATUS_SETTLE_S)                           # the enable status read right after connecting is stale
        if not self._all_enabled():
            if self._estopped():
                raise RuntimeError('controller in emergency stop: support the arm and clear it in the console first (resume is never sent here)')
            deadline = time.monotonic() + 3.; confirmed = 0
            while confirmed < 3:                              # three fresh all-enabled reads in a row
                if time.monotonic() > deadline:
                    raise RuntimeError(f'not all joints enabled within 3 s: {list(self.sdk.GetArmEnableStatus())}')
                self.sdk.EnableArm(7); time.sleep(.15); confirmed = confirmed + 1 if self._all_enabled() else 0
        self.firmware_limits = self._read_angle_limits(); self.gripper_ready = False
        return hardware_limits(self.firmware_limits)

    def _all_enabled(self):
        status = self.sdk.GetArmEnableStatus(); return len(status) == 6 and all(status)

    def _estopped(self):
        try:
            return int(self.sdk.GetArmStatus().arm_status.arm_status) == 0x01
        except AttributeError:
            return False

    def _read_angle_limits(self):
        out = []
        for joint in range(1, 7):
            previous = float(self.sdk.GetCurrentMotorAngleLimitMaxVel().time_stamp)
            self.sdk.SearchMotorMaxAngleSpdAccLimit(joint, 0x01)
            deadline = time.monotonic() + .5; item = None
            while time.monotonic() < deadline:
                msg = self.sdk.GetCurrentMotorAngleLimitMaxVel(); motor = msg.current_motor_angle_limit_max_vel
                if float(msg.time_stamp) > 0 and float(msg.time_stamp) != previous and motor.motor_num == joint:
                    lo, hi = (getattr(motor, k, 0x7FFF) for k in ('min_angle_limit', 'max_angle_limit'))   # 0.1 deg; 0x7FFF = unset
                    item = None if 0x7FFF in (lo, hi) or not lo < hi else tuple(np.deg2rad([lo / 10, hi / 10]))
                    break
                time.sleep(.01)
            out.append(item)
        return out

    def joints(self):
        """Measured joints (6,) in rad."""
        msg = self.sdk.GetArmJointMsgs()
        return np.deg2rad(np.array([getattr(msg.joint_state, f'joint_{i}') for i in range(1, 7)], float) / 1000)

    def gripper_width(self):
        """Measured gripper opening in m."""
        return float(self.sdk.GetArmGripperMsgs().gripper_state.grippers_angle) / 1e6

    def send_joints(self, q):
        """Joint target (6,) rad, as returned by PiperIK.tick(). Call once per control cycle."""
        units = np.rint(np.rad2deg(np.asarray(q, float)) * 1000).astype(int)
        self.sdk.MotionCtrl_2(0x01, 0x01, int(self.speed_percent), 0x00)
        self.sdk.JointCtrl(*map(int, units))

    def send_gripper(self, width_m, effort_raw=GRIPPER_EFFORT_RAW):
        if not self.gripper_ready:                           # official order: clear errors (0x02), enable (0x01), at the current opening
            w = int(round(max(0., self.gripper_width()) * 1e6))
            self.sdk.GripperCtrl(w, int(effort_raw), 0x02, 0x00); time.sleep(.02)
            self.sdk.GripperCtrl(w, int(effort_raw), 0x01, 0x00); self.gripper_ready = True
        self.sdk.GripperCtrl(int(round(width_m * 1e6)), int(effort_raw), 0x01, 0x00)

    def disable(self):
        """Motors off (the arm sinks under its brakes / damping: support it or put it in a safe pose first)."""
        self.sdk.DisableArm(7)

    def disconnect(self):
        if self.sdk is not None:
            self.sdk.DisconnectPort(); self.sdk = None
