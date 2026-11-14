"""FastUMI-100K pose convention <-> base-frame TCP poses, identical to data_viewer/robot.py (alignment 'fastumi').

FastUMI-100K per arm, 7 numbers: [x, y, z, rx, ry, rz, g]
  x y z     metres, fingertip TCP position
  rx ry rz  radians, scipy Rotation.from_euler('xyz', ...) / as_euler('xyz'), wrapped to +-pi like the dataset
  g         gripper opening 0..1 (0 = closed, 1 = open)
Positions and rotations share one frame: the gripper at the episode's first frame, x forward (approach), y left,
z up. The first frame is all zeros and action[i] = state[i+1]: actions are absolute poses in that frame, not deltas.
"""
import numpy as np
from scipy.spatial.transform import Rotation

# Columns: Piper tool0 axes (x down, y finger axis left, z forward) written in FastUMI's initial-gripper frame
# (x forward, y left, z up). Positions use the same frame (checked 2026-10-09 on 303 grasps / releases).
FASTUMI_ROT_TO_TOOL = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
FASTUMI_POS_TO_TOOL = FASTUMI_ROT_TO_TOOL.T
PIPER_TCP_AXES = ('down', 'left', 'forward')                 # = FASTUMI_ROT_TO_TOOL in words
_DIRECTIONS = {'forward': (1., 0., 0.), 'back': (-1., 0., 0.), 'left': (0., 1., 0.), 'right': (0., -1., 0.), 'up': (0., 0., 1.), 'down': (0., 0., -1.)}


def tcp_axes_matrix(axes=None):
    """Where the TCP frame's x, y, z axes point on a gripper held level and straight ahead, as three words from
    forward / back / left / right / up / down (FastUMI gripper frame: x forward, y left, z up), or as the 3x3 matrix
    whose columns are those directions. None: the Piper's tool0 = ('down', 'left', 'forward')."""
    if axes is None:
        return FASTUMI_ROT_TO_TOOL
    if len(axes) == 3 and all(isinstance(a, str) for a in axes):
        if any(a not in _DIRECTIONS for a in axes):
            raise ValueError(f'tcp axes are three of {list(_DIRECTIONS)}, got {axes}')
        A = np.array([_DIRECTIONS[a] for a in axes]).T
    else:
        A = np.array(axes, float)
    if A.shape != (3, 3) or not np.allclose(A.T @ A, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(A), 1., atol=1e-9):
        raise ValueError(f'tcp axes must form a right-handed frame (x cross y = z), got {axes}')
    return A

# Gripper: data 0 -> 0 mm, 1 -> 70 mm, effort 1000 (0.001 N*m units; 300 cannot move the FastUMI gripper).
GRIPPER_ZERO_M, GRIPPER_ONE_M, GRIPPER_EFFORT_RAW = 0., .07, 1000


def row_to_matrix(row):
    """[x, y, z, rx, ry, rz] (m, rad, scipy 'xyz') -> 4x4."""
    row = np.asarray(row, float); T = np.eye(4)
    T[:3, 3] = row[:3]; T[:3, :3] = Rotation.from_euler('xyz', row[3:6]).as_matrix()
    return T


def fastumi_to_robot(rows, anchor, first=None, axes=None):
    """FastUMI rows (N, >=6) -> (N, 4, 4) base-frame TCP targets.

    anchor: the base-frame TCP pose (4x4) that FastUMI pose `first` sits at; `first` defaults to all zeros, the
    episode start. axes: the TCP frame's axes (tcp_axes_matrix; default the Piper's). Same formula as the console:
    anchor @ [A.T R0.T R A, C R0.T (p - p0)].
    """
    rows = np.atleast_2d(np.asarray(rows, float)); F = row_to_matrix(np.zeros(6) if first is None else first)
    A, C = (FASTUMI_ROT_TO_TOOL, FASTUMI_POS_TO_TOOL) if axes is None else (axes, axes.T); out = []
    for row in rows:
        T = row_to_matrix(row); local = np.eye(4)
        local[:3, :3] = A.T @ F[:3, :3].T @ T[:3, :3] @ A
        local[:3, 3] = C @ F[:3, :3].T @ (T[:3, 3] - F[:3, 3])
        out.append(anchor @ local)
    return np.array(out)


def robot_to_fastumi(pose, anchor, first=None, axes=None):
    """Inverse of fastumi_to_robot for one pose: base-frame TCP (4x4) -> FastUMI [x, y, z, rx, ry, rz]."""
    F = row_to_matrix(np.zeros(6) if first is None else first)
    A, C = (FASTUMI_ROT_TO_TOOL, FASTUMI_POS_TO_TOOL) if axes is None else (axes, axes.T)
    local = np.linalg.inv(anchor) @ np.asarray(pose, float)
    R = F[:3, :3] @ A @ local[:3, :3] @ A.T
    p = F[:3, 3] + F[:3, :3] @ C.T @ local[:3, 3]
    return np.r_[p, Rotation.from_matrix(R).as_euler('xyz')]


def gripper_width(g, zero_m=GRIPPER_ZERO_M, one_m=GRIPPER_ONE_M):
    """Data gripper value -> commanded width (m). The calibrated data range is 0..1; a value outside it is sent
    at the nearest end (the console refuses such values; a policy can overshoot slightly)."""
    g = float(np.clip(g, 0., 1.))
    return (1 - g) * zero_m + g * one_m


def gripper_value(width_m, zero_m=GRIPPER_ZERO_M, one_m=GRIPPER_ONE_M):
    """Measured width (m) -> data gripper value (not clipped)."""
    return (float(width_m) - zero_m) / (one_m - zero_m)


def episode_rows(state, action, timestamps):
    """One arm of a recorded episode (7 columns each) -> (rows, times) as the console replays it:
    rows = [state[0], action[0], ..., action[N-2]] (row i = pose at data time i), times from 0 s."""
    state = np.asarray(state, float); action = np.asarray(action, float); t = np.asarray(timestamps, float).reshape(-1)
    return np.vstack([state[0], action[:-1]]), t - t[0]
