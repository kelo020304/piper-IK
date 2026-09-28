"""Independent third-order position / SO(3) low-pass filter."""
import numpy as np
from scipy.spatial.transform import Rotation
from ._validation import pose_matrix, positive


class ThirdOrderPoseFilter:
    """Three cascaded first-order pose filters, with per-stage time constant tau.

    Input/output: absolute 4x4 poses in the same fixed frame, metres/radians.
    Rotation updates follow the shortest SO(3) arc. This is a reference filter,
    not a joint trajectory generator or a velocity/acceleration guarantee.
    Instances retain state; use one per stream. Not thread-safe.
    """

    def __init__(self, initial_pose, tau=0.06, *, linear_speed=None, angular_speed=None):
        self.tau = positive(tau, 'tau')
        self.linear_speed = None if linear_speed is None else positive(linear_speed, 'linear_speed')
        self.angular_speed = None if angular_speed is None else positive(angular_speed, 'angular_speed')
        self.reset(initial_pose)

    def reset(self, pose):
        """Set all three stages to pose, clearing the previous filter state."""
        pose = pose_matrix(pose)
        self._stages = [pose.copy() for _ in range(3)]

    @property
    def value(self):
        """Return a copy of the current filtered pose."""
        return self._stages[-1].copy()

    def update(self, target, dt):
        """Advance by dt seconds; target must be an absolute pose."""
        source = pose_matrix(target, 'target')
        dt = positive(dt, 'dt')
        alpha = -np.expm1(-dt / self.tau)
        for stage in self._stages:
            dp = alpha * (source[:3, 3] - stage[:3, 3])
            length = np.linalg.norm(dp)
            if self.linear_speed is not None and length > self.linear_speed * dt:
                dp *= self.linear_speed * dt / length
            rv = alpha * Rotation.from_matrix(source[:3, :3] @ stage[:3, :3].T).as_rotvec()
            angle = np.linalg.norm(rv)
            if self.angular_speed is not None and angle > self.angular_speed * dt:
                rv *= self.angular_speed * dt / angle
            stage[:3, 3] += dp
            stage[:3, :3] = Rotation.from_rotvec(rv).as_matrix() @ stage[:3, :3]
            source = stage
        return self.value
