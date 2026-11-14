"""Check a new arm's URDF before using it: what the IK takes from it and how well the IK solves on it.

    python -m piper_ik.check path/to/arm.urdf [--tcp-frame tool0] [--ik-frame LINK] [--joints j1,j2,...]
                             [--tcp-axes down,left,forward] [--q0 a1,a2,...] [--n 200]

Prints the arm joints and their limits, the locked joints, the IK frame, the TCP (offset from the IK frame), where the
TCP axes point at q0, and two IK round trips over random postures inside the limits: seeds 0.05 rad from the answer
(the 200 Hz case: the previous cycle's solution) and seeds 0.6 rad away. Exit status 1 when fewer than 95% of the near
seeds converge.
"""
import argparse
import sys
import numpy as np
from .core import ArmIK

# base axis -> FastUMI word for a gripper that points along base +x, level (FastUMI x forward, y left, z up)
BASE_WORDS = {(1, 0, 0): ('+x', 'forward'), (-1, 0, 0): ('-x', 'back'), (0, 1, 0): ('+y', 'left'), (0, -1, 0): ('-y', 'right'),
              (0, 0, 1): ('+z', 'up'), (0, 0, -1): ('-z', 'down')}


def _axis(v):
    """(base axis, FastUMI word, degrees off) of the base axis a unit vector is closest to."""
    i = int(np.argmax(np.abs(v))); key = tuple(int(c) for c in np.eye(3)[i] * np.sign(v[i]))
    return BASE_WORDS[key] + (float(np.degrees(np.arccos(np.clip(abs(v[i]), -1, 1)))),)


def round_trip(ik, n, noise, rng):
    lo, hi = ik.joint_limits; pad = .05 * (hi - lo); rows = []
    for _ in range(n):
        q = rng.uniform(lo + pad, hi - pad); seed = np.clip(q + rng.normal(size=ik.dof) * noise, lo, hi)
        r = ik.solve(ik.fk(q), seed); rows.append((r.converged, r.position_error * 1e3, np.degrees(r.orientation_error), r.seconds * 1e3))
    a = np.array(rows, float)
    return a[:, 0].mean(), np.percentile(a[:, 1], 95), a[:, 1].max(), np.percentile(a[:, 2], 95), a[:, 3].mean()


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m piper_ik.check', description=__doc__.split('\n\n')[0])
    ap.add_argument('urdf'); ap.add_argument('--tcp-frame', default='tool0'); ap.add_argument('--ik-frame')
    ap.add_argument('--joints', help='comma-separated arm joint names (default: the chain from the root to the TCP frame)')
    ap.add_argument('--tcp-axes', help='three words, e.g. down,left,forward (the Piper default)')
    ap.add_argument('--q0', help='comma-separated joints for the axes printout (default: middle of the limits)')
    ap.add_argument('--n', type=int, default=200, help='postures per round trip'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    ik = ArmIK(a.urdf, tcp_frame=a.tcp_frame, ik_frame=a.ik_frame, joints=a.joints.split(',') if a.joints else None,
               tcp_axes=tuple(a.tcp_axes.split(',')) if a.tcp_axes else None)
    lo, hi = ik.joint_limits; m = ik._k.model
    print(f'URDF        {a.urdf}')
    print(f'arm joints  {ik.dof}:')
    for i, name in enumerate(ik.joint_names):
        prismatic = m.joints[i + 1].shortname().startswith(('JointModelP', 'JointModelPrismatic'))
        unit, f = ('m', 1.) if prismatic else ('deg', np.degrees(1.))
        print(f'  {name:<24} {"prismatic" if prismatic else "revolute":<10} {lo[i] * f:9.3f} .. {hi[i] * f:9.3f} {unit}')
    print(f'locked      {locked_joints(a.urdf, ik.joint_names) or "none"} (held at 0)')
    print(f'IK frame    {ik.ik_frame}')
    t = ik.tcp[:3, 3] * 1e3
    print(f'TCP         {a.tcp_frame}: {np.linalg.norm(t):.1f} mm from {ik.ik_frame} (xyz {t[0]:.1f} {t[1]:.1f} {t[2]:.1f} mm)')
    q0 = np.array([float(v) for v in a.q0.split(',')]) if a.q0 else (lo + hi) / 2
    T = ik.fk(q0)
    axes = [_axis(T[:3, i]) for i in range(3)]
    print(f'at q0       q0 = {np.round(q0, 3).tolist()}; TCP at {np.round(T[:3, 3], 3).tolist()} m; TCP axes in the base frame: '
          + ', '.join(f'{c} {b} ({off:.0f} deg off)' for c, (b, _, off) in zip('xyz', axes)))
    if max(off for *_, off in axes) < 1:
        print(f'            tcp_axes = {",".join(w for _, w, _ in axes)} if the gripper at q0 is level and points along base +x '
              '(otherwise pick such a q0 with --q0)')
    words = {tuple(np.eye(3)[i] * s): w for i, ws in enumerate((('forward', 'back'), ('left', 'right'), ('up', 'down'))) for s, w in zip((1, -1), ws)}
    used = [words.get(tuple(np.round(ik.tcp_axes[:, i], 9) + 0.), 'tilted') for i in range(3)]
    print(f'tcp_axes    in use: TCP x {used[0]}, y {used[1]}, z {used[2]} on a level gripper (FastUMI convention)')
    rng = np.random.default_rng(a.seed); ok = True
    for label, noise in (('near seeds (0.05 rad)', .05), ('far seeds (0.6 rad)', .6)):
        c, p95, mx, r95, ms = round_trip(ik, a.n, noise, rng)
        print(f'IK {label:<22} converged {c * 100:5.1f}%  position p95 {p95:.4f} mm, max {mx:.3f} mm  rotation p95 {r95:.4f} deg  {ms:.2f} ms/solve')
        ok = ok and (noise > .1 or c >= .95)
    print('OK' if ok else 'FAIL: fewer than 95% of the near-seed solves converged (check limits, TCP frame and joints)')
    return 0 if ok else 1


def locked_joints(urdf, arm):
    """Movable joints of the URDF that are not arm joints (locked at 0)."""
    from .kinematics import pin
    full = pin.buildModelFromUrdf(str(urdf))
    return [n for n in list(full.names)[1:] if n not in arm]


if __name__ == '__main__':
    sys.exit(main())
