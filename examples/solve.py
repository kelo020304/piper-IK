"""python examples/solve.py --backend pinocchio (or rl)."""
import argparse
import numpy as np
from piper_ik import PiperIK

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['pinocchio', 'rl'], default='pinocchio')
args = parser.parse_args()
ik = PiperIK(backend=args.backend)
q = np.array([0., 1., -1.2, .2, .55, 0.])
target = ik.fk(q)
target[0, 3] += .005
result = ik.solve(target, seed=q)
print('q [rad]:', result.q)
print(f'converged={result.converged}, position={result.position_error*1000:.3f} mm, '
      f'orientation={np.rad2deg(result.orientation_error):.3f} deg')
