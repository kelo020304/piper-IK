"""ArmIK: one arm from its URDF in one object. It holds the model, FK / IK, and the data_viewer real-arm replay
chain (timestamp_interpolation_ik_direct_v2) that turns FastUMI actions into joint commands. PiperIK is ArmIK on the
bundled Piper + UMI gripper model (model/piper_umi.urdf).

Every control cycle, tick():
  1. advance data time by the real elapsed time x the 追点 time scale (follow.py; 1 in 'strict' mode),
  2. sample the target at that data time (linear position + Slerp between the two neighbouring points),
  3. IK seeded with the previous cycle's solution,
  4. return that solution; the caller sends it to the arm unchanged.
No pose filter, no joint planner, no velocity / acceleration limit, no rescaling of the trajectory.
"""
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import ARM_JOINTS, KinematicModel, SolverConfig, solve_flange
from .convention import fastumi_to_robot, robot_to_fastumi, gripper_width, gripper_value, tcp_axes_matrix, GRIPPER_ZERO_M, GRIPPER_ONE_M
from .follow import Follow, key_times
from .trajectory import Trajectory

MODEL = Path(__file__).resolve().parent / 'model' / 'piper_umi.urdf'
CONTROL_HZ = 200        # the arm's own rate: it reports its joints every 5 ms
CONTROL_DT = 1. / CONTROL_HZ
DATA_FPS = 20.          # FastUMI-100K frame rate: action k of a chunk is the pose (k+1)/20 s after its observation
PUSH_HISTORY_S = 30.    # push_actions keeps this much trajectory before t_obs (grasp / release detection); older points are never used

# Piper datasheet ranges (deg); the hand-measured J5 range -76.8..+73.9 agrees.
PIPER_SPEC_LIMITS_DEG = ([-154., 0., -175., -100., -75., -170.], [154., 195., 0., 112., 75., 170.])
# piper_sdk joint table: what the arm really reaches with the UMI gripper (J6 stalls at -122.8 deg although the
# firmware reports +-180).
PIPER_SDK_TABLE_DEG = ([-150., 0., -170., -100., -70., -120.], [150., 179.9, 0., 100., 70., 120.])


def hardware_limits(firmware=None):
    """Joint bounds (rad): datasheet ∩ SDK table, narrowed by the controller's firmware readback where given
    (six (lo, hi) pairs in rad or None, as PiperArm.enable() reads them). The bundled URDF carries hardware_limits()."""
    spec = np.deg2rad(PIPER_SPEC_LIMITS_DEG); table = np.deg2rad(PIPER_SDK_TABLE_DEG)
    lo = np.maximum(spec[0], table[0]); hi = np.minimum(spec[1], table[1])
    for i, item in enumerate(firmware or []):
        if item is not None:
            lo[i] = max(lo[i], item[0]); hi[i] = min(hi[i], item[1])
    return lo, hi


@dataclass
class IKResult:
    """Joints plus residuals recomputed at the TCP."""
    q: np.ndarray
    converged: bool
    position_error: float       # m
    orientation_error: float    # rad
    seconds: float


def _pose(T):
    T = np.array(T, float)
    R = T[:3, :3] if T.shape == (4, 4) else None
    if R is None or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8, rtol=0) \
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(R), 1, atol=1e-6, rtol=0):
        raise ValueError('pose must be a finite 4x4 homogeneous matrix with a proper rotation')
    return T


class ArmIK:
    """One arm from its URDF. Kinematics: fk(q), solve(target, seed). Execution of FastUMI actions: start(),
    observation(), push_actions(), tick(). Thread-safe: tick() in the control loop, observation() / push_actions()
    from the inference thread. Use a new instance (or start() again) for every episode.

    urdf: the arm (fixed base; revolute / prismatic arm joints with limits). tcp_frame: the fingertip TCP frame in it.
    ik_frame / joints / tcp: see kinematics.KinematicModel (defaults: found from the URDF).
    tcp_axes: where the TCP frame's x, y, z point on a level gripper (convention.tcp_axes_matrix); default the Piper's
        ('down', 'left', 'forward').
    limits: (lo, hi) rad, e.g. what the arm's driver reads; default: the URDF's.
    follow: 'wait' (追点, default) or 'strict' (1x data time). fps: policy / dataset frame rate.
    """

    def __init__(self, urdf, limits=None, *, tcp_frame='tool0', ik_frame=None, joints=None, tcp=None, tcp_axes=None,
                 follow='wait', fps=DATA_FPS, gripper_zero_m=GRIPPER_ZERO_M, gripper_one_m=GRIPPER_ONE_M, config=SolverConfig()):
        self._lock = threading.RLock()                       # Pinocchio Data is shared by fk / solve: one caller at a time
        self._k = KinematicModel(urdf, ik_frame, tcp_frame, joints, tcp); self.config = config
        self.tcp = self._k.tcp.copy(); self._tcp_inv = np.linalg.inv(self.tcp)
        self.urdf = Path(urdf); self.joint_names = self._k.joint_names; self.dof = len(self.joint_names)
        self.ik_frame = self._k.model.frames[self._k.frame].name
        self._axes = None if tcp_axes is None else tcp_axes_matrix(tcp_axes)   # None: the Piper's, the exact module constants
        self.set_joint_limits(*((self._k.lower.copy(), self._k.upper.copy()) if limits is None else limits))
        self.fps = float(fps); self._follow = Follow(follow); self.zero_m, self.one_m = float(gripper_zero_m), float(gripper_one_m)
        self._anchor = None; self._traj = None; self._keys = []
        self._measured = None; self._q = None; self._seed = None; self._g0 = None
        self.source_time = 0.; self.last_ik = None; self.target = None

    # ---------------------------------------------------------------- kinematics
    @property
    def joint_limits(self):
        return self._k.lower.copy(), self._k.upper.copy()

    @property
    def tcp_axes(self):
        return tcp_axes_matrix(self._axes).copy()

    def set_joint_limits(self, lower, upper):
        """IK bounds in rad, written into the solver's own arrays (as data_viewer apply_joint_limits does)."""
        lo, hi = np.array(lower, float).reshape(-1), np.array(upper, float).reshape(-1)
        if lo.shape != (self.dof,) or hi.shape != (self.dof,) or np.any(lo > hi):
            raise ValueError(f'limits must be two {self.dof}-vectors in rad with lower <= upper')
        with self._lock:
            self._k.lower[:] = lo; self._k.upper[:] = hi

    def _joints(self, q, check=True):
        q = np.array(q, float).reshape(-1)
        if q.shape != (self.dof,) or not np.isfinite(q).all():
            raise ValueError(f'joints must be {self.dof} finite values (rad / m)')
        if check and (np.any(q < self._k.lower) or np.any(q > self._k.upper)):
            raise ValueError('joints exceed the joint limits')
        return q

    def fk(self, q):
        """Base-frame TCP pose (4x4) of joints q. Measured joints may sit slightly outside the limits: not checked."""
        q = self._joints(q, check=False)
        with self._lock:
            return self._k.fk(q).homogeneous.copy() @ self.tcp

    def solve(self, target, seed):
        """IK for a base-frame TCP target (4x4) from seed joints inside the limits. Returns the best solution
        inside the limits even when the target is out of reach (converged=False, residuals set)."""
        return self._solve(_pose(target), seed)

    def _solve(self, target, seed):
        with self._lock:
            seed = self._joints(seed); t0 = time.perf_counter()
            q = self._joints(solve_flange(self._k, target @ self._tcp_inv, seed, self.config))
            T = self.fk(q)
        pe = float(np.linalg.norm(T[:3, 3] - target[:3, 3]))
        oe = float(Rotation.from_matrix(T[:3, :3].T @ target[:3, :3]).magnitude())
        return IKResult(q, pe < self.config.position_tolerance and oe < self.config.orientation_tolerance, pe, oe, time.perf_counter() - t0)

    # ---------------------------------------------------------------- execution of FastUMI actions
    def start(self, q_measured, gripper_width_m, q_command=None, anchor=None):
        """Episode start, arm standing still. FastUMI state 0 (the policy's first frame) = the TCP pose here.
        q_command: joints last sent to the arm, if known (it sags a little below its command); default: measured.
        anchor: base-frame TCP pose for FastUMI state 0; default FK(q_command). Only a recorded replay with a
        planned placement passes it; for a policy keep the default."""
        with self._lock:
            m = self._joints(q_measured, check=False); q0 = m if q_command is None else self._joints(q_command, check=False)
            self._seed = np.clip(q0, self._k.lower, self._k.upper); self._q = self._seed.copy(); self._measured = m.copy()
            self._anchor = self.fk(self._seed) if anchor is None else _pose(anchor); self.target = self._anchor.copy()
            self._g0 = gripper_value(gripper_width_m, self.zero_m, self.one_m)
            self._traj = None; self._keys = []; self.source_time = 0.; self.last_ik = None
            self._follow.reset(m, self._q)

    def push_actions(self, actions, t_obs):
        """A policy chunk: actions (H, 7) [x y z rx ry rz g], absolute FastUMI values exactly as in the dataset's
        `action` column (un-normalized, not deltas), predicted from the observation taken at data time t_obs (the
        t_obs that observation() returned with it). Action k is the target at t_obs + (k+1)/fps; it replaces
        everything the previous chunk planned after t_obs. Points more than PUSH_HISTORY_S before t_obs are dropped,
        so a chunk costs the same at minute 30 as at minute 1. The new plan is built outside the lock: tick() keeps
        running meanwhile and switches to it at its next cycle."""
        with self._lock:
            self._started()
        a = np.atleast_2d(np.asarray(actions, float))
        if a.ndim != 2 or a.shape[1] != 7 or len(a) < 1 or not np.isfinite(a).all():
            raise ValueError('actions must be a finite (H, 7) array: x y z rx ry rz g')
        t_obs = float(t_obs)
        while True:
            with self._lock:
                self._started(); base, anchor, g0 = self._traj, self._anchor, self._g0
            traj, keys = self._spliced(base, anchor, g0, a, t_obs)
            with self._lock:
                if self._traj is base and self._anchor is anchor:       # nothing replaced the plan meanwhile
                    self._traj, self._keys = traj, keys
                    return

    def _spliced(self, base, anchor, g0, a, t_obs):
        base = base if base is not None else Trajectory([0.], [anchor], [g0])
        i = max(0, int(np.searchsorted(base.times, t_obs - PUSH_HISTORY_S, side='right')) - 1)
        if i:
            base = Trajectory(base.times[i:], base.poses[i:], base.grip[i:])
        traj = base.splice(t_obs, t_obs + np.arange(1, len(a) + 1) / self.fps, fastumi_to_robot(a[:, :6], anchor, axes=self._axes), a[:, 6])
        return traj, key_times(traj.grip, traj.times)

    def _set_plan(self, traj, keys):
        """Install a trajectory built by an identical ArmIK's push_actions (RealtimeArm's control process)."""
        with self._lock:
            self._started(); self._traj = traj; self._keys = list(keys)

    def load_trajectory(self, rows, times, keys=None):
        """A whole recorded trajectory instead of chunks: rows (N, 7) FastUMI poses with rows[0] at the start pose,
        times (N,) s (see convention.episode_rows). keys: grasp / release times (s from rows[0]); default: detected."""
        with self._lock:
            self._started()
            rows = np.asarray(rows, float); t = np.asarray(times, float) - float(times[0])
            self._traj = Trajectory(t, fastumi_to_robot(rows[:, :6], self._anchor, first=rows[0], axes=self._axes), rows[:, 6])
            self._keys = key_times(rows[:, 6], t) if keys is None else [float(k) for k in keys]
            self.source_time = 0.

    def tick(self, q_measured, dt):
        """One control cycle. q_measured: joints now (rad); dt: real seconds since the previous tick (monotonic).
        Returns (q_command (dof,) rad, gripper width m): send both to the arm exactly as returned."""
        with self._lock:
            self._started()
            m = self._joints(q_measured, check=False); before = self._measured if self._measured is not None else m; self._measured = m.copy()
            raw = float(dt); cdt = float(np.clip(raw, .001, .04))
            if self._traj is None:
                target, g = self._anchor, self._g0                       # no chunk yet: hold the start pose
            else:
                if self.source_time < self._traj.end:
                    scale = self._follow.update(cdt, self._q, m, before, self.source_time, self._keys)
                    self.source_time = min(self._traj.end, self.source_time + scale * max(0., raw))
                target, g = self._traj.at(self.source_time)
            result = self._solve(target, self._seed)                # target: a fresh pose from the trajectory
            self._q = result.q.copy(); self._seed = self._q.copy(); self.last_ik = result; self.target = target
            return self._q.copy(), gripper_width(g, self.zero_m, self.one_m)

    def observation(self, q_measured, gripper_width_m):
        """(state (7,), t_obs): the measured arm as a FastUMI observation.state row and the data time it belongs to.
        Pass t_obs back to push_actions() with the chunk computed from this observation."""
        with self._lock:
            self._started()
            return np.r_[robot_to_fastumi(self.fk(q_measured), self._anchor, axes=self._axes), gripper_value(gripper_width_m, self.zero_m, self.one_m)], float(self.source_time)

    def wait_until(self, t, poll_s=.002, stop=None):
        """Block until data time reaches t (e.g. t_obs + replan_steps / fps before the next inference), the last
        point is reached, or the threading.Event `stop` is set."""
        while self.source_time < t and not self.finished and not (stop is not None and stop.is_set()):
            time.sleep(poll_s)

    def to_robot(self, rows):
        """FastUMI rows (N, >=6) -> base-frame TCP poses of this episode (after start())."""
        self._started(); return fastumi_to_robot(rows, self._anchor, axes=self._axes)

    def to_fastumi(self, pose):
        """Base-frame TCP pose -> FastUMI [x y z rx ry rz] of this episode (after start())."""
        self._started(); return robot_to_fastumi(pose, self._anchor, axes=self._axes)

    def _started(self):
        if self._anchor is None:
            raise RuntimeError('call start() first')

    # ---------------------------------------------------------------- read-only status
    @property
    def finished(self):
        """True when data time has reached the last point pushed so far (the arm holds it)."""
        return self._traj is not None and self.source_time >= self._traj.end

    @property
    def end_time(self):
        return None if self._traj is None else self._traj.end

    @property
    def follow_scale(self):
        return self._follow.scale

    @property
    def anchor(self):
        return None if self._anchor is None else self._anchor.copy()


class PiperIK(ArmIK):
    """ArmIK on the bundled Piper + UMI gripper (model/piper_umi.urdf): IK frame link6, TCP tool0 = model/tcp.json,
    limits = hardware_limits(). Bit-identical to the data_viewer console.

    limits: (lo, hi) rad, e.g. PiperArm.enable(); default: the URDF's (= hardware_limits())."""

    def __init__(self, limits=None, *, follow='wait', fps=DATA_FPS, gripper_zero_m=GRIPPER_ZERO_M, gripper_one_m=GRIPPER_ONE_M,
                 urdf=MODEL, config=SolverConfig()):
        super().__init__(urdf, limits, tcp_frame='tool0', ik_frame='link6', joints=ARM_JOINTS, follow=follow, fps=fps, gripper_zero_m=gripper_zero_m,
                         gripper_one_m=gripper_one_m, config=config)
