"""The standalone pose filter in front of plain IK, for a pose stream outside the execution chain (for example a
teleoperation target). No device, no simulator. RealtimeArm / tick() never filter: see README section 10.

    python piper_ik/examples/filter_target.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the folder that contains piper_ik/
from piper_ik import PiperIK, ThirdOrderPoseFilter

ik = PiperIK()
q = np.array([0., 1., -1.2, .2, .55, 0.])
initial = ik.fk(q)
pose_filter = ThirdOrderPoseFilter(initial, tau=.06)
target = initial.copy()
target[0, 3] += .005                                                # a 5 mm step of the target

for _ in range(100):
    reference = pose_filter.update(target, dt=.02)                  # 50 Hz stream
    result = ik.solve(reference, q)
    if not result.converged:
        raise RuntimeError(f'IK residual: {result.position_error} m, {result.orientation_error} rad')
    q = result.q                                                    # warm start for the next sample
print(f'final reference error: {np.linalg.norm(ik.fk(q)[:3, 3] - target[:3, 3]) * 1e3:.4f} mm')
