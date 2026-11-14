"""Template: a FastUMI-trained policy driving the real Piper through piper_ik at 200 Hz.

Fill in `Policy.infer()` and `grab_images()`. Leave everything else as it is. RealtimeArm runs the control loop in a
process of its own (every 5 ms: read the arm, PiperIK.tick(), send exactly what it returns); this script is the
policy loop and never touches the arm directly:

    python piper_ik/examples/vla_loop.py --can can0 --seconds 60
    python piper_ik/examples/vla_loop.py --can sim --seconds 10        # the same loops against the simulated arm

Before running on hardware: CAN up (sudo ip link set can0 type can bitrate 1000000; sudo ip link set can0 up), the arm
already standing in the pose the episode starts from (move it there with your own reset first).
"""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the folder that contains piper_ik/
from piper_ik import RealtimeArm

REPLAN_STEPS = 10      # run the policy again after this many data frames (20 fps) of the current chunk


class Policy:
    def infer(self, images, state):
        """images: your camera frames; state: (7,) FastUMI observation.state [x y z rx ry rz g].
        Return (H, 7) actions in the dataset's own units: absolute (not deltas), un-normalized, euler 'xyz' rad."""
        raise NotImplementedError


def grab_images():
    raise NotImplementedError          # read your cameras now; the state below is read right after


def run(arm, policy, grab_images, seconds, replan_steps=REPLAN_STEPS):
    """The policy loop. `arm` is a started RealtimeArm. Returns after `seconds`; an error here (or in the control
    process) ends it, and the arm holds its last command."""
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        images = grab_images()
        state, t_obs = arm.observation()
        actions = policy.infer(images, state)
        arm.push_actions(actions, t_obs)                     # t_obs: the data time of this observation, from observation()
        arm.wait_until(t_obs + replan_steps / arm.fps)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--can', default='can0'); ap.add_argument('--seconds', type=float, default=60.); a = ap.parse_args()
    with RealtimeArm(a.can) as arm:                          # connect, enable (this arm's joint limits), start at the current pose (FastUMI state 0)
        run(arm, Policy(), grab_images, a.seconds)           # afterwards the arm keeps holding the last command, still enabled


if __name__ == '__main__':
    main()
