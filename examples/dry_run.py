"""No hardware: the same loop as vla_loop.py, with a simulated arm and a fake policy whose chunks follow a known
path (forward / sideways / down, a pitch change, a grasp and a release). Prints how closely the fingertip stays on
that path:

    python piper_ik/examples/dry_run.py                  # 追点 ('wait', default)
    python piper_ik/examples/dry_run.py --follow strict
"""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the folder that contains piper_ik/
from piper_ik import PiperIK, CONTROL_DT
from piper_ik.sim import SimArm

Q_START = np.array([0., 1.0, -1.2, 0., .6, 0.])     # gripper ahead of the base, pointing forward and down
H, REPLAN, T_END = 16, 4, 8.                        # chunk length, replan every 4 data frames, path length (s)


def path(t):
    """The fake policy's intention, a FastUMI row [x y z rx ry rz g] at data time t (all zeros at t = 0)."""
    t = float(np.clip(t, 0., T_END)); s = np.sin(np.pi * t / T_END)
    g = 1. if t < 3. else (0. if t < 6. else 1.)                                  # grasp at 3 s, release at 6 s
    return np.array([.05 * s, .04 * np.sin(2 * np.pi * t / T_END), -.03 * s, 0., np.deg2rad(15) * s, 0., g])


def policy(state, t_obs, fps):
    """Stands in for the VLA: (H, 7) absolute FastUMI actions, action k for data time t_obs + (k+1)/fps."""
    return np.array([path(t_obs + (k + 1) / fps) for k in range(H)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--follow', default='wait', choices=['wait', 'strict']); a = ap.parse_args()
    arm = SimArm(Q_START); ik = PiperIK(follow=a.follow)
    ik.start(arm.joints(), arm.gripper_width())
    dense = ik.to_robot(np.array([path(t)[:6] for t in np.arange(0, T_END, .002)]))[:, :3, 3]
    next_inference = 0.; wall = 0.; off = []; ik_mm = []; ik_deg = []; chunks = 0
    while ik.source_time < T_END and wall < 120:
        # In the real loop this is the inference thread (see vla_loop.py); here it runs when data time reaches the replan point.
        if ik.source_time >= next_inference and ik.source_time < T_END:
            state, t_obs = ik.observation(arm.joints(), arm.gripper_width())
            ik.push_actions(policy(state, t_obs, ik.fps), t_obs); chunks += 1; next_inference = t_obs + REPLAN / ik.fps
        q_cmd, width = ik.tick(arm.joints(), CONTROL_DT)                           # one control cycle
        arm.send_joints(q_cmd); arm.send_gripper(width)                              # sent exactly as returned
        arm.step(CONTROL_DT); wall += CONTROL_DT
        tip = ik.fk(arm.joints())[:3, 3]; off.append(np.min(np.linalg.norm(dense - tip, axis=1)) * 1e3)
        ik_mm.append(ik.last_ik.position_error * 1e3); ik_deg.append(np.degrees(ik.last_ik.orientation_error))
    off = np.array(off)
    print(f'follow={a.follow}: {T_END:.1f} s of data in {wall:.1f} s ({wall / T_END:.2f}x), {chunks} chunks of {H} pushed, replan every {REPLAN} frames')
    print(f'IK residual max {max(ik_mm):.3f} mm / {max(ik_deg):.3f} deg')
    print(f'fingertip distance to the policy path: p95 {np.percentile(off, 95):.2f} mm, max {off.max():.2f} mm')


if __name__ == '__main__':
    main()
