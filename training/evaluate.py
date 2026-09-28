"""Independent random reachable pose evaluation through the public SDK."""
import argparse
import json
from pathlib import Path
import numpy as np
from piper_ik import PiperIK, PinocchioConfig, RLConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ik', choices=['rl', 'pinocchio'], default='rl')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=94027)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--out', type=Path, default=Path('artifacts/evaluation.json'))
    args = parser.parse_args()
    if args.samples <= 0 or args.iterations <= 0:
        parser.error('samples and iterations must be positive')
    ik = PiperIK(args.ik,
        pinocchio=PinocchioConfig(max_iterations=args.iterations, position_tolerance=.003,
                                 orientation_tolerance=float(np.deg2rad(1.5))),
        rl=RLConfig(weights=args.weights, max_iterations=args.iterations, time_budget_s=None))
    lo, hi = ik.joint_limits
    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.samples):
        q = rng.uniform(lo+.02, hi-.02)
        target = ik.fk(rng.uniform(lo+.02, hi-.02))
        result = ik.solve(target, q)
        rows.append(dict(position_m=result.position_error, orientation_rad=result.orientation_error,
                         seconds=result.seconds, success=result.converged))
    report = dict(ik=args.ik, samples=args.samples, seed=args.seed, iterations=args.iterations,
        success_rate=float(np.mean([r['success'] for r in rows])),
        position_median_m=float(np.median([r['position_m'] for r in rows])),
        orientation_median_deg=float(np.rad2deg(np.median([r['orientation_rad'] for r in rows]))),
        scope='Random FK targets and seeds; local IK, 3mm/1.5deg; no dynamics or collision checks.', cases=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}, indent=2))


if __name__ == '__main__':
    main()
