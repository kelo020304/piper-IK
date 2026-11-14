"""Time-indexed TCP targets, sampled exactly like data_viewer ActionSchedule ('frame' mode).

At data time t: find the two neighbouring points, position linear, rotation Slerp (shortest path), gripper value
linear. No filter, no spline, no averaging: every point is passed through and corners stay corners.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate(a, b, k):
    """Pose between 4x4 a (k=0) and b (k=1): linear position, Slerp rotation."""
    out = np.eye(4); k = float(np.clip(k, 0, 1)); out[:3, 3] = (1 - k) * a[:3, 3] + k * b[:3, 3]
    out[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))([k]).as_matrix()[0]
    return out


class Trajectory:
    """Points (time s, 4x4 base-frame TCP pose, gripper value 0..1) with strictly increasing times."""

    def __init__(self, times, poses, grip):
        self.times = np.asarray(times, float).reshape(-1); self.poses = np.asarray(poses, float); self.grip = np.asarray(grip, float).reshape(-1)
        if len(self.times) < 1 or self.poses.shape != (len(self.times), 4, 4) or self.grip.shape != self.times.shape:
            raise ValueError('times, poses (N,4,4) and grip must have the same length N >= 1')
        if np.any(np.diff(self.times) <= 0):
            raise ValueError('trajectory times must be strictly increasing')
        self._segment = (-1, -1, None)                           # (i, j, Slerp of poses i -> j): built once per segment

    @property
    def end(self):
        return float(self.times[-1])

    def at(self, t):
        """(pose 4x4, gripper value) at data time t; before the first / after the last point it holds that point."""
        t = float(np.clip(t, self.times[0], self.times[-1]))
        idx = max(0, min(len(self.times) - 1, int(np.searchsorted(self.times, t, side='right') - 1)))
        end = min(idx + 1, len(self.times) - 1)
        k = 0. if end == idx else (t - self.times[idx]) / (self.times[end] - self.times[idx])
        kk = float(np.clip(k, 0, 1))
        return self._interpolate(idx, end, k), float((1 - kk) * self.grip[idx] + kk * self.grip[end])

    def _interpolate(self, i, j, k):
        """interpolate(poses[i], poses[j], k), bit for bit, reusing the segment's Slerp: at 200 Hz ten cycles share one
        20 fps segment."""
        a, b = self.poses[i], self.poses[j]; seg = self._segment
        if seg[0] != i or seg[1] != j:
            seg = self._segment = (i, j, Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))))
        out = np.eye(4); k = float(np.clip(k, 0, 1)); out[:3, 3] = (1 - k) * a[:3, 3] + k * b[:3, 3]
        out[:3, :3] = seg[2]([k]).as_matrix()[0]
        return out

    def splice(self, t0, times, poses, grip):
        """Keep the points before t0, insert this trajectory's own target at t0, then the new points (all after t0).
        This is how a new action chunk replaces the rest of the old one."""
        pose0, g0 = self.at(t0); keep = self.times < t0 - 1e-9
        return Trajectory(np.r_[self.times[keep], t0, times], np.concatenate([self.poses[keep], pose0[None], poses]),
                          np.r_[self.grip[keep], g0, grip])
