"""Shared input validation; never silently normalizes an invalid pose."""
import numpy as np


def pose_matrix(value, name='pose'):
    value = np.array(value, dtype=float, copy=True)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite 4x4 homogeneous matrix')
    R = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-6, rtol=0)):
        raise ValueError(f'{name} must contain a proper rotation and homogeneous last row')
    return value


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)
