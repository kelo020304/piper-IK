"""Compose filtering and IK; no device or simulator connection."""
import numpy as np
from piper_ik import PiperIK, ThirdOrderPoseFilter

ik = PiperIK()
q = np.array([0., 1., -1.2, .2, .55, 0.])
initial = ik.fk(q)
pose_filter = ThirdOrderPoseFilter(initial, tau=.06)
target = initial.copy()
target[0, 3] += .005

for _ in range(100):
    reference = pose_filter.update(target, dt=.02)
    result = ik.solve(reference, seed=q)
    if not result.converged:
        raise RuntimeError(f'IK residual: {result.position_error} m, {result.orientation_error} rad')
    q = result.q  # Warm start for the next sample, not a motor command.
print('Final reference error [m]:', np.linalg.norm(ik.fk(q)[:3, 3]-target[:3, 3]))
