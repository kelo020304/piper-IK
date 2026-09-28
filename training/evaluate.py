"""CPU, ROS-free held-out FK targets; no CAN, dynamics, or robot commands."""
import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_ik.policy import Policy
from piper_ik.pin_solver import PinIK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ik', choices=['rl', 'pinocchio'], default='rl')
    parser.add_argument('--weights', type=Path, default=ROOT/'assets/weights/ppo_precision.pt')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=94027)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/evaluation.json')
    args = parser.parse_args()
    if args.samples <= 0 or args.iterations <= 0:
        parser.error('samples and iterations must be positive')
    model = mujoco.MjModel.from_xml_path(str(ROOT/'assets/models/piper.xml'))
    data = mujoco.MjData(model)
    tip = model.body('tool0').id
    lo, hi = model.jnt_range.T
    def fk(q):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        pose = np.eye(4)
        pose[:3, 3] = data.xpos[tip]
        pose[:3, :3] = data.xmat[tip].reshape(3, 3)
        return pose
    solver = Policy(args.weights, lo, hi) if args.ik == 'rl' else PinIK()
    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.samples):
        q = rng.uniform(lo+.02, hi-.02)
        target = fk(rng.uniform(lo+.02, hi-.02))
        started = time.perf_counter()
        if args.ik == 'pinocchio':
            q = solver.solve(target, q, max_iter=args.iterations).q
        else:
            for _ in range(args.iterations):
                pose = fk(q)
                pe = np.linalg.norm(pose[:3, 3]-target[:3, 3])
                oe = Rotation.from_matrix(pose[:3, :3].T@target[:3, :3]).magnitude()
                if pe < .003 and oe < np.deg2rad(1.5):
                    break
                q = solver.propose(q, pose, target)
        elapsed = time.perf_counter()-started
        pose = fk(q)
        pe = float(np.linalg.norm(pose[:3, 3]-target[:3, 3]))
        oe = float(Rotation.from_matrix(pose[:3, :3].T@target[:3, :3]).magnitude())
        rows.append({'position_m': pe, 'orientation_rad': oe, 'seconds': elapsed,
                     'success': bool(pe < .003 and oe < np.deg2rad(1.5))})
    report = {'ik': args.ik, 'samples': args.samples, 'seed': args.seed,
              'iterations': args.iterations, 'success_rate': float(np.mean([r['success'] for r in rows])),
              'position_median_m': float(np.median([r['position_m'] for r in rows])),
              'orientation_median_deg': float(np.rad2deg(np.median([r['orientation_rad'] for r in rows]))),
              'scope': 'Independent random reachable tool0 poses; local IK from random starts; no restarts, smoothing, dynamics, collisions or hardware. Not the historical 256-case validation set.',
              'cases': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
