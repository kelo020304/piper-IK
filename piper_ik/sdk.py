"""Small, backend-independent Piper FK / IK API."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ._validation import pose_matrix, positive


@dataclass(frozen=True)
class PinocchioConfig:
    weights: tuple = (1., 1., 1., .3, .3, .3)
    damping: float = 1e-7
    singularity_threshold: float = .01
    seed_weight: float = 1e-8
    max_joint_step: float = .2
    max_iterations: int = 40
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-3

    def __post_init__(self):
        _validate_solver_config(self)
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (6,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('weights must contain six finite positive values')
        object.__setattr__(self, 'weights', tuple(weights.tolist()))
        positive(self.damping, 'damping')
        positive(self.max_joint_step, 'max_joint_step')
        for name in ('singularity_threshold', 'seed_weight'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class RLConfig:
    weights: str | Path | None = None
    max_iterations: int = 20
    time_budget_s: float | None = .008
    position_tolerance: float = .003
    orientation_tolerance: float = float(np.deg2rad(1.5))

    def __post_init__(self):
        _validate_solver_config(self)
        if self.time_budget_s is not None:
            positive(self.time_budget_s, 'time_budget_s')


def _validate_solver_config(config):
    if isinstance(config.max_iterations, bool) or not isinstance(config.max_iterations, int) or config.max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    positive(config.position_tolerance, 'position_tolerance')
    positive(config.orientation_tolerance, 'orientation_tolerance')


@dataclass
class IKResult:
    """Absolute joints plus independently recomputed TCP residuals."""
    q: np.ndarray
    converged: bool
    position_error: float  # metres
    orientation_error: float  # radians
    seconds: float


class IKBackend(Protocol):
    """Custom solver: base-frame absolute tool0 target + seed -> absolute joints.

    Inputs are private copies. Return a finite six-vector within joint limits.
    The wrapper handles TCP conversion, validation and residual reporting.
    """
    def solve(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray: ...


class _RLBackend:
    def __init__(self, model, config):
        from ._policy import Policy
        self.model, self.config = model, config
        weights = config.weights or Path(__file__).resolve().parent / 'assets/ppo_precision.pt'
        self.policy = Policy(weights, model.lower, model.upper)

    def solve(self, target, seed):
        c = self.config
        started, q = time.perf_counter(), seed.copy()
        for _ in range(c.max_iterations):
            current = self.model.fk(q).homogeneous
            pe, oe = _errors(current, target)
            if pe < c.position_tolerance and oe < c.orientation_tolerance:
                break
            if c.time_budget_s is not None and time.perf_counter() - started >= c.time_budget_s:
                break
            q = self.policy.propose(q, current, target)
        return q


def _errors(current, target):
    return (float(np.linalg.norm(current[:3, 3] - target[:3, 3])),
            float(Rotation.from_matrix(current[:3, :3].T @ target[:3, :3]).magnitude()))


class PiperIK:
    """FK / IK in the base frame, metres and radians. No I/O or motion execution.

    backend: 'pinocchio', 'rl', or an IKBackend object.
    tcp: fixed 4x4 tool0-to-TCP transform (identity by default).
    solve() requires an explicit seed; reuse result.q for continuous targets.
    Instances own mutable FK scratch data and are not thread-safe.
    """
    def __init__(self, backend: str | IKBackend = 'pinocchio', *, tcp=None,
                 pinocchio: PinocchioConfig | None = None, rl: RLConfig | None = None,
                 urdf=None, frame='tool0'):
        from ._pinocchio import KinematicModel, PinocchioBackend
        self._model = KinematicModel(urdf, frame)
        self._tcp = pose_matrix(np.eye(4) if tcp is None else tcp, 'tcp')
        if isinstance(backend, str):
            if backend == 'pinocchio':
                config = pinocchio or PinocchioConfig()
                self._backend = PinocchioBackend(self._model, config)
            elif backend == 'rl':
                if urdf is not None or frame != 'tool0':
                    raise ValueError('RL uses the packaged Piper model and tool0 frame; use tcp for tool offsets')
                config = rl or RLConfig()
                self._backend = _RLBackend(self._model, config)
            else:
                raise ValueError("backend must be 'pinocchio', 'rl', or an IKBackend object")
        else:
            if not callable(getattr(backend, 'solve', None)):
                raise TypeError('custom backend must implement solve(target, seed)')
            config = pinocchio or PinocchioConfig()
            self._backend = backend
        self.position_tolerance = config.position_tolerance
        self.orientation_tolerance = config.orientation_tolerance

    @property
    def joint_limits(self):
        """Return independent (lower, upper) six-vectors in radians."""
        return self._model.lower.copy(), self._model.upper.copy()

    def _joints(self, q, *, check_limits=True):
        q = np.array(q, dtype=float, copy=True)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('q must contain six finite joint angles in radians')
        if check_limits and (np.any(q < self._model.lower) or np.any(q > self._model.upper)):
            raise ValueError('q exceeds model joint limits')
        return q

    def fk(self, q, *, check_limits=True):
        """Return absolute base-frame TCP pose (4x4).

        Use check_limits=False only to observe measured joints, which may cross
        model limits through encoder quantization or simulator soft contacts.
        Values are never clipped; shape/finite checks and IK limits still apply.
        """
        return self._model.fk(self._joints(q, check_limits=check_limits)).homogeneous.copy() @ self._tcp

    def solve(self, target, seed):
        """Solve an absolute TCP target from six absolute seed angles.

        Invalid inputs raise ValueError; an unmet tolerance returns
        converged=False and the residual, without executing any motion.
        """
        target, seed = pose_matrix(target, 'target'), self._joints(seed)
        started = time.perf_counter()
        flange_target = target @ np.linalg.inv(self._tcp)
        q = self._joints(self._backend.solve(flange_target, seed))
        pe, oe = _errors(self.fk(q), target)
        return IKResult(q, pe < self.position_tolerance and oe < self.orientation_tolerance,
                        pe, oe, time.perf_counter() - started)
