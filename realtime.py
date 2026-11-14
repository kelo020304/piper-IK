"""RealtimeArm: the arm, its driver and ArmIK.tick() in a control process of their own, at CONTROL_HZ (200 Hz).

Your policy process only asks for observations and hands over action chunks; cameras, the GPU and any Python work
there never delay a control cycle. The control process runs one loop on an absolute 5 ms schedule:
    driver.read(): the joint / gripper feedback that has arrived (Piper: raw SocketCAN, kernel-filtered to 0x2A5-0x2A8)
    -> ArmIK.tick(measured joints, real elapsed time)
    -> driver.send(): exactly what it returns (Piper: 0x151 + 0x155-0x157 + 0x159, the bytes piper_sdk would send).
push_actions() builds the new trajectory in this process, in an identical ArmIK (same model, limits, start pose and
chunks, hence the same trajectory), and only the finished trajectory crosses to the control process.
observation() reads the latest feedback and data time from shared memory.

    with RealtimeArm('can0') as arm:            # connect, enable (this arm's joint limits), start at the current pose
        state, t_obs = arm.observation()
        arm.push_actions(policy.infer(images, state), t_obs)
        arm.wait_until(t_obs + 10 / arm.fps)
RealtimeArm('sim') runs the same control process against the simulated Piper of sim.py. Another arm: pass its model
(robot=dict(urdf=..., tcp_frame=...), the ArmIK arguments) and its driver (see Driver).
"""
import multiprocessing
import time
import traceback
from dataclasses import dataclass
import numpy as np
from .core import ArmIK, PiperIK, CONTROL_HZ, DATA_FPS, hardware_limits
from .arm import PiperArm, PiperCAN, command_frames, gripper_frame
from .convention import GRIPPER_EFFORT_RAW
from .trajectory import Trajectory

# shared state, one double each (a six-joint arm; Layout(n) for n joints)
SEQ, ALIVE, TIME, FINISHED, END, SCALE, Q, WIDTH, CMD, CMD_WIDTH, IK_PE, IK_OE, IK_OK, CYCLES, LATE, DROPPED = \
    0, 1, 2, 3, 4, 5, 6, 12, 13, 19, 20, 21, 22, 23, 24, 25
SIZE = 26


class Layout:
    """Indices of the shared array for an n-joint arm: measured joints Q:Q+n, commanded joints CMD:CMD+n."""

    def __init__(self, n):
        self.n = n; self.SEQ, self.ALIVE, self.TIME, self.FINISHED, self.END, self.SCALE, self.Q = range(7)
        self.WIDTH = self.Q + n; self.CMD = self.WIDTH + 1; self.CMD_WIDTH = self.CMD + n
        self.IK_PE, self.IK_OE, self.IK_OK, self.CYCLES, self.LATE, self.DROPPED = range(self.CMD_WIDTH + 1, self.CMD_WIDTH + 7)
        self.SIZE = self.DROPPED + 1


assert vars(Layout(6)) == dict(n=6, SEQ=SEQ, ALIVE=ALIVE, TIME=TIME, FINISHED=FINISHED, END=END, SCALE=SCALE, Q=Q, WIDTH=WIDTH, CMD=CMD,
                               CMD_WIDTH=CMD_WIDTH, IK_PE=IK_PE, IK_OE=IK_OE, IK_OK=IK_OK, CYCLES=CYCLES, LATE=LATE, DROPPED=DROPPED, SIZE=SIZE)


@dataclass
class LastIK:
    position_error: float       # m, at the TCP
    orientation_error: float    # rad
    converged: bool


def _connect(can, speed_percent):
    """(joint limits, PiperCAN with feedback). On can0: enable through piper_sdk as PiperArm does, then close the SDK
    (its reader thread parses every frame on the bus) and keep only a kernel-filtered raw socket."""
    if can.startswith('unix:'):
        io = PiperCAN.unix(can[5:]); limits = hardware_limits()
    else:
        arm = PiperArm(can); arm.connect(); limits = arm.enable(speed_percent=speed_percent)
        arm.sdk.DisconnectPort()
        io = PiperCAN.socketcan(can)
    deadline = time.monotonic() + 1.
    while True:
        io.poll()
        if io.ready:
            return limits, io
        if time.monotonic() > deadline:
            raise RuntimeError(f'no joint / gripper feedback on {can} within 1 s')
        time.sleep(.005)


class Driver:
    """What RealtimeArm's control process needs from an arm; write one of these for an arm other than the Piper.
    It is made in the policy process and runs in the control process, so it must be picklable (a module-level class
    with plain attributes; open hardware in open(), not in __init__). No method may block for long: read() and send()
    run every 5 ms.

      open()          connect and enable the arm, standing still. Returns (limits, q0, width0): the joint limits for
                      the IK ((lo, hi) rad, or None for the URDF's), the joints now (rad, the URDF's joint order) and
                      the gripper opening now (m)
      read()          (joints, gripper width m) latest feedback, without waiting
      send(q, width)  one cycle's command, exactly as given (no filter, no limit)
      close()         release the hardware (the arm keeps its last command)
      dropped / received   optional counters for RealtimeArm's statistics
    """
    dropped = received = 0

    def open(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def send(self, q, width):
        raise NotImplementedError

    def close(self):
        pass


class PiperDriver(Driver):
    """The Piper on SocketCAN ('can0', ...) or on the simulated bus ('unix:<address>'): enable through piper_sdk, then a
    raw socket; each cycle the five frames PiperArm.send_joints() + send_gripper() send (MOVE J at speed_percent)."""

    def __init__(self, can='can0', speed_percent=100, effort_raw=GRIPPER_EFFORT_RAW):
        self.can, self.speed_percent, self.effort_raw = can, speed_percent, effort_raw; self.io = None

    def open(self):
        limits, self.io = _connect(self.can, self.speed_percent)
        q0, w0 = self.io.joints(), self.io.gripper_width()
        w = max(0., w0)                                                     # official gripper order: clear errors (0x02), enable (0x01)
        self.io.send([gripper_frame(w, self.effort_raw, 0x02)]); time.sleep(.02); self.io.send([gripper_frame(w, self.effort_raw, 0x01)])
        return limits, q0, w0

    def read(self):
        self.io.poll(); return self.io.joints(), self.io.gripper_width()

    def send(self, q, width):
        self.io.send(command_frames(q, width, self.speed_percent, self.effort_raw))

    def close(self):
        if self.io is not None:
            self.io.close()

    dropped = property(lambda self: self.io.dropped)
    received = property(lambda self: self.io.received)


def make_ik(robot, limits, follow):
    """The control process's and the planner's ArmIK: robot = ArmIK keyword arguments (urdf, tcp_frame, ...), or None
    for the bundled Piper (PiperIK)."""
    return PiperIK(limits, follow=follow) if robot is None else ArmIK(limits=limits, follow=follow, **robot)


def _control(conn, shared, driver, robot, follow, control_hz, keep_stats):
    """The control process."""
    io = None
    try:
        ik = make_ik(robot, None, follow); L = Layout(ik.dof)               # the model first: the first cycle follows open() at once
        io = driver; limits, q0, w0 = driver.open()
        if limits is not None:
            ik.set_joint_limits(*limits)
        ik.start(q0, w0)
        conn.send(('ready', {'limits': ik.joint_limits, 'q0': q0, 'w0': w0, 'anchor': ik.anchor}))
        period = 1. / control_hz; nxt = last = time.monotonic(); cycles = late = 0; stats = ([], [], []) if keep_stats else None
        while True:
            now = time.monotonic()
            while conn.poll():
                msg = conn.recv()
                if msg[0] == 'traj':
                    ik._set_plan(Trajectory(msg[1], msg[2], msg[3]), msg[4])
                elif msg[0] == 'stop':
                    conn.send(('stats', None if stats is None else {'interval_s': np.diff(stats[0]), 'cycle_s': np.array(stats[1]), 'ik_position_error_m': np.array(stats[2]),
                                                                    'dropped_frames': io.dropped, 'received_frames': io.received}))
                    return
            q, w = io.read()
            q_cmd, width = ik.tick(q, now - last); last = now
            io.send(q_cmd, width)
            r = ik.last_ik; cycles += 1
            with shared.get_lock():
                shared[L.SEQ] = cycles; shared[L.TIME] = ik.source_time; shared[L.FINISHED] = float(ik.finished)
                shared[L.END] = ik.end_time if ik.end_time is not None else -1.; shared[L.SCALE] = ik.follow_scale
                shared[L.Q:L.Q + L.n] = np.asarray(q, float).tolist(); shared[L.WIDTH] = w
                shared[L.CMD:L.CMD + L.n] = q_cmd.tolist(); shared[L.CMD_WIDTH] = width
                shared[L.IK_PE] = r.position_error; shared[L.IK_OE] = r.orientation_error; shared[L.IK_OK] = float(r.converged)
                shared[L.CYCLES] = cycles; shared[L.LATE] = late; shared[L.DROPPED] = io.dropped; shared[L.ALIVE] = 1.
            if stats is not None:
                stats[0].append(now); stats[1].append(time.monotonic() - now); stats[2].append(r.position_error)
            nxt += period
            if time.monotonic() > nxt:
                late += 1
                if time.monotonic() > nxt + period:                         # more than a period behind: restart the schedule
                    nxt = time.monotonic()
            time.sleep(max(0., nxt - time.monotonic()))
    except (EOFError, BrokenPipeError):                                     # the policy process is gone: stop sending, the arm holds
        pass
    except BaseException:
        try:
            conn.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        shared[ALIVE] = 0.                                                  # index 1 for every layout
        if io is not None:
            try:
                io.close()
            except Exception:                                               # a driver that failed half-way through open()
                pass


class RealtimeArm:
    """The real Piper (can='can0', ...) or the simulated one (can='sim') driven at control_hz from its own process.
    Another arm: robot = its ArmIK keyword arguments (dict(urdf=..., tcp_frame=..., ...)) and driver = its Driver;
    `can`, speed_percent and effort_raw are then not used.
    Use as a context manager, or start() / stop(). One per arm and episode: the pose at start() is FastUMI state 0."""

    def __init__(self, can='can0', *, follow='wait', control_hz=CONTROL_HZ, speed_percent=100, effort_raw=GRIPPER_EFFORT_RAW,
                 keep_stats=False, start_timeout=20., sim_q0=None, robot=None, driver=None):
        if type(speed_percent) is not int or not 1 <= speed_percent <= 100:
            raise ValueError('speed_percent must be an int 1..100 (use 100)')
        self.can, self.follow, self.control_hz, self.speed_percent, self.effort_raw = can, follow, float(control_hz), speed_percent, effort_raw
        self.keep_stats, self.start_timeout, self.sim_q0 = keep_stats, start_timeout, sim_q0
        self.robot = None if robot is None else dict(robot); self.driver = driver
        self._L = Layout(6) if self.robot is None else Layout(make_ik(self.robot, None, follow).dof)
        if driver is None and self._L.n != 6:
            raise ValueError(f'a {self._L.n}-joint arm needs its own driver (RealtimeArm(robot=..., driver=...)); the default drives a Piper')
        self.fps = float(self.robot.get('fps', DATA_FPS)) if self.robot else DATA_FPS
        self.sim = None; self._proc = None; self._plan = None; self._error = None; self.stats = None

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        driver = self.driver
        if driver is None:
            can = self.can
            if can == 'sim':
                from .sim import SimBus, SIM_Q0
                self.sim = SimBus(SIM_Q0 if self.sim_q0 is None else self.sim_q0); can = 'unix:' + self.sim.address
            driver = PiperDriver(can, self.speed_percent, self.effort_raw)
        ctx = multiprocessing.get_context('spawn')                          # safe with CUDA / threads in this process
        self._conn, child = ctx.Pipe(); self._shared = ctx.Array('d', self._L.SIZE)
        self._proc = ctx.Process(target=_control, name='piper_ik control', daemon=True,
                                 args=(child, self._shared, driver, self.robot, self.follow, self.control_hz, self.keep_stats))
        self._proc.start(); child.close()
        if not self._conn.poll(self.start_timeout):
            self._proc.terminate(); raise RuntimeError(f'control process gave no answer within {self.start_timeout:.0f} s')
        kind, info = self._conn.recv()
        if kind == 'error':
            self._proc.join(1.); raise RuntimeError('control process failed to start:\n' + info)
        self._plan = make_ik(self.robot, info['limits'], self.follow); self._plan.start(info['q0'], info['w0'])
        if not np.array_equal(self._plan.anchor, info['anchor']):
            raise RuntimeError('planner and control process disagree on the start pose')
        self.joint_limits = info['limits']
        while self._shared[self._L.CYCLES] < 1:                             # first cycle done
            self._state()
            time.sleep(.001)
        return self

    def stop(self):
        """Stop the loop (the arm holds its last command, still enabled). Returns the cycle stats when keep_stats."""
        if self._proc is not None and self._proc.is_alive():
            try:
                self._conn.send(('stop',))
                if self._conn.poll(2.):
                    kind, info = self._conn.recv()
                    self.stats = info if kind == 'stats' else self.stats
            except (OSError, EOFError):
                pass
            self._proc.join(2.)
            if self._proc.is_alive():
                self._proc.terminate()
        if self.sim is not None:
            self.sim.close(); self.sim = None
        return self.stats

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---------------------------------------------------------------- policy side
    def observation(self):
        """(state (7,), t_obs): the measured arm as a FastUMI observation.state row and the data time it belongs to.
        Pass t_obs back to push_actions() with the chunk computed from this observation."""
        L = self._L; v = self._state(); state, _ = self._plan.observation(v[L.Q:L.Q + L.n], v[L.WIDTH])
        return state, float(v[L.TIME])

    def push_actions(self, actions, t_obs):
        """Same contract as PiperIK.push_actions: (H, 7) absolute FastUMI actions predicted from observation t_obs."""
        self._state(); self._plan.push_actions(actions, t_obs); tr = self._plan._traj
        self._conn.send(('traj', tr.times, tr.poses, tr.grip, list(self._plan._keys)))

    def wait_until(self, t, poll_s=.002):
        """Block until data time reaches t or the last pushed point."""
        while True:
            v = self._state()
            if v[self._L.TIME] >= t or v[self._L.FINISHED] > .5:
                return
            time.sleep(poll_s)

    # ---------------------------------------------------------------- status
    def _state(self):
        with self._shared.get_lock():
            v = np.array(self._shared[:])
        if v[self._L.ALIVE] < .5 and v[self._L.CYCLES] > 0 or self._proc is None or not self._proc.is_alive():
            if self._conn.poll():
                try:
                    kind, info = self._conn.recv()
                    self._error = info if kind == 'error' else self._error
                except (OSError, EOFError):
                    pass
            raise RuntimeError('the control process stopped; the arm holds its last command' + (':\n' + self._error if self._error else ''))
        return v

    @property
    def source_time(self): return float(self._state()[self._L.TIME])

    @property
    def finished(self): return bool(self._state()[self._L.FINISHED] > .5)

    @property
    def follow_scale(self): return float(self._state()[self._L.SCALE])

    @property
    def cycles(self): return int(self._state()[self._L.CYCLES])

    @property
    def late_cycles(self): return int(self._state()[self._L.LATE])

    @property
    def last_ik(self):
        L = self._L; v = self._state(); return LastIK(float(v[L.IK_PE]), float(v[L.IK_OE]), bool(v[L.IK_OK] > .5))

    @property
    def joints(self):
        """Latest measured joints (dof,) rad."""
        L = self._L; return self._state()[L.Q:L.Q + L.n].copy()

    @property
    def command(self):
        """(joints (dof,) rad, gripper width m) sent in the latest cycle."""
        L = self._L; v = self._state(); return v[L.CMD:L.CMD + L.n].copy(), float(v[L.CMD_WIDTH])

    def fk(self, q):
        return self._plan.fk(q)
