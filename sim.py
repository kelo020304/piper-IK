"""Simulated arms for runs without hardware.

SimArm: each joint chases its latest target on its own, acceleration-limited, the way the Piper drive behaves. Any
number of joints.
SimBus: a six-joint SimArm behind a Unix socket that speaks the Piper's CAN frames (struct can_frame, the same bytes as
can0): it decodes the joint and gripper commands it receives and, every 5 ms like the real arm, sends the joint and
gripper feedback (0x2A5-0x2A8, the frames the kernel filter lets through on can0). RealtimeArm('sim') drives one.
SimDriver: a SimArm of any size as a RealtimeArm driver (no sockets), to try another arm's model before its driver.
"""
import itertools
import os
import socket
import struct
import threading
import time
import numpy as np

CAN_FRAME = struct.Struct('=IB3x8s')
SIM_Q0 = np.array([0., 1.0, -1.2, 0., .6, 0.])        # gripper ahead of the base, pointing forward and down
_count = itertools.count()


class SimArm:
    """Each joint chases its latest target on its own, acceleration-limited: how the Piper drive behaves."""
    def __init__(self, q, amax=5., vmax=3.):
        self.q = np.array(q, float); self.v = np.zeros(len(self.q)); self.goal = self.q.copy(); self.amax, self.vmax = amax, vmax; self.width = .07

    def joints(self): return self.q.copy()
    def gripper_width(self): return self.width
    def send_joints(self, q): self.goal = np.array(q, float)
    def send_gripper(self, width_m): self.width = width_m

    def step(self, dt):
        err = self.goal - self.q
        want = np.clip(np.sign(err) * np.sqrt(2 * self.amax * np.abs(err)), -self.vmax, self.vmax)
        self.v += np.clip(want - self.v, -self.amax * dt, self.amax * dt)
        arrive = np.abs(self.v * dt) >= np.abs(err)
        self.q = np.where(arrive, self.goal, self.q + self.v * dt); self.v = np.where(arrive, 0., self.v)


class SimBus:
    """SimArm on a Unix SEQPACKET socket (address: self.address, for PiperCAN.unix / RealtimeArm(can='unix:...')).
    Runs in a thread of the process that creates it. commands: CAN frames received per id."""

    def __init__(self, q0=SIM_Q0, width=.07, period=.005):
        self.address = f'\0piper_ik_sim_{os.getpid()}_{next(_count)}'
        self.arm = SimArm(q0); self.arm.width = width; self.period = period; self.commands = {}
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET); self._server.bind(self.address); self._server.listen(1)
        self._stop = threading.Event(); self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()

    def _run(self):
        self._server.settimeout(.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            conn.setblocking(False); nxt = time.monotonic()
            try:
                while not self._stop.is_set():
                    while True:
                        try:
                            cf = conn.recv(CAN_FRAME.size)
                        except BlockingIOError:
                            break
                        if not cf:
                            raise ConnectionError
                        cid, _, data = CAN_FRAME.unpack(cf); self.commands[cid] = self.commands.get(cid, 0) + 1
                        if 0x155 <= cid <= 0x157:
                            j = 2 * (cid - 0x155); self.arm.goal[j:j + 2] = np.radians(np.array(struct.unpack('>ii', data), float) / 1000)
                        elif cid == 0x159:
                            self.arm.width = struct.unpack('>i', data[:4])[0] / 1e6
                    self.arm.step(self.period); q = self.arm.q
                    for j in range(3):
                        u = [int(round(np.degrees(v) * 1000)) for v in q[2 * j:2 * j + 2]]
                        conn.send(CAN_FRAME.pack(0x2A5 + j, 8, struct.pack('>ii', *u)))
                    conn.send(CAN_FRAME.pack(0x2A8, 8, struct.pack('>ihBB', int(round(self.arm.width * 1e6)), 0, 0, 0)))
                    nxt += self.period; time.sleep(max(0., nxt - time.monotonic()))
            except (ConnectionError, BrokenPipeError, OSError):
                pass
            finally:
                conn.close()

    def close(self):
        self._stop.set(); self._thread.join(timeout=1.); self._server.close()


class SimDriver:
    """RealtimeArm(driver=SimDriver(q0), robot=...): a SimArm with len(q0) joints inside the control process. Each
    read() advances it by the real time since the previous read. limits: (lo, hi) rad for the IK, None = the URDF's."""
    dropped = received = 0

    def __init__(self, q0, width=.07, limits=None, amax=5., vmax=3.):
        self.q0 = np.array(q0, float); self.width, self.limits, self.amax, self.vmax = width, limits, amax, vmax; self.arm = None

    def open(self):
        self.arm = SimArm(self.q0, self.amax, self.vmax); self.arm.width = self.width; self._t = time.monotonic()
        return self.limits, self.arm.joints(), self.arm.gripper_width()

    def read(self):
        now = time.monotonic(); self.arm.step(now - self._t); self._t = now
        return self.arm.joints(), self.arm.gripper_width()

    def send(self, q, width):
        self.arm.send_joints(q); self.arm.send_gripper(width)

    def close(self):
        pass
