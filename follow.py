"""追点 / "wait for the arm" pacing, identical to data_viewer runtime.py (_follow_scale, FOLLOW_*).

The Piper drive chases every new joint target on its own: each joint independently, acceleration-limited. A 1x
stream the arm cannot keep up with makes the joints fall out of step and the fingertip leave the path. So data time
advances only as fast as the arm keeps up: lead = how far the last command is ahead of the measured joints. Full
speed while the lead is <= 0.25 deg, data time stops at 1 deg (0.1 / 0.4 deg within 0.3 s of a grasp or release).

Only time stretches. The path, every target point and every command stay exactly as computed; nothing is filtered.
"""
import numpy as np

FOLLOW_LEAD_DEG = (.25, 1.)        # full speed at <= 0.25 deg lead; data time stops at 1 deg
FOLLOW_KEY_LEAD_DEG = (.1, .4)     # tighter within FOLLOW_KEY_WINDOW_S of a grasp / release moment
FOLLOW_KEY_WINDOW_S = .3
FOLLOW_SPEEDUP_PER_S = 4.          # the time scale rises at most this much per second; it drops at once
FOLLOW_STILL_RAD_S = .02           # arm standing still but 'behind' (static sag): keep creeping forward
FOLLOW_CREEP = .1
SAG_CLIP_DEG = 3.                  # static offset (gravity sag) measured at the start, counted out of the lead


class Follow:
    """mode 'wait' (default, 跟不上就等) or 'strict' (always 1x data time)."""

    def __init__(self, mode='wait'):
        if mode not in ('wait', 'strict'):
            raise ValueError("follow mode must be 'wait' or 'strict'")
        self.mode = mode; self.scale = 1.; self.offset = np.zeros(6)

    def reset(self, measured, command):
        """At the start, standing still: measured - command is the arm's static sag."""
        self.scale = 1.; self.offset = np.clip(np.asarray(measured, float) - np.asarray(command, float), -np.deg2rad(SAG_CLIP_DEG), np.deg2rad(SAG_CLIP_DEG))

    def update(self, dt, command, measured, before, source_time, key_times):
        """How fast data time may advance this cycle (1 = recorded speed).
        dt: this cycle (s, clipped to 1..40 ms); command: the previous cycle's command; measured / before: joints now
        and one cycle ago (rad); key_times: grasp / release data times."""
        if self.mode != 'wait':
            self.scale = 1.; return 1.
        lead = float(np.max(np.abs(np.asarray(command) + self.offset - np.asarray(measured))))
        near = any(abs(source_time - k) <= FOLLOW_KEY_WINDOW_S for k in key_times)
        lo, hi = np.deg2rad(FOLLOW_KEY_LEAD_DEG if near else FOLLOW_LEAD_DEG)
        want = float(np.clip((hi - lead) / (hi - lo), 0., 1.))
        if want < FOLLOW_CREEP and float(np.max(np.abs(np.asarray(measured) - np.asarray(before)))) / max(dt, 1e-3) < FOLLOW_STILL_RAD_S:
            want = FOLLOW_CREEP
        self.scale = want if want < self.scale else min(want, self.scale + FOLLOW_SPEEDUP_PER_S * dt)
        return self.scale


# Grasp / release moments (data_viewer replay_metrics.py, unchanged): zigzag with hysteresis on the 0..1 gripper value.
PHASE_MIN_CHANGE = .06   # a closing / opening must move the gripper value at least this much
SETTLE_FRACTION = .1     # motion starts / settles within this fraction of the phase's change


def gripper_phases(g, times, h=PHASE_MIN_CHANGE):
    g = np.asarray(g, float); n = len(g)
    if n < 2:
        return []
    piv, trend, lo, hi, ext = [], 0, 0, 0, 0
    for i in range(1, n):
        if trend == 0:
            hi = i if g[i] > g[hi] else hi; lo = i if g[i] < g[lo] else lo
            if g[hi] - g[lo] >= h:
                first, ext, trend = (hi, lo, -1) if hi < lo else (lo, hi, 1)
                piv.append(first)
        elif trend > 0:
            if g[i] > g[ext]:
                ext = i
            elif g[ext] - g[i] >= h:
                piv.append(ext); ext, trend = i, -1
        else:
            if g[i] < g[ext]:
                ext = i
            elif g[i] - g[ext] >= h:
                piv.append(ext); ext, trend = i, 1
    if trend:
        piv.append(ext)
    phases = []
    for a, b in zip(piv[:-1], piv[1:]):
        change = float(g[b] - g[a]); tol = max(.01, SETTLE_FRACTION * abs(change)); seg = g[a:b + 1]
        still = np.flatnonzero(np.abs(seg - g[a]) <= tol); settled = np.flatnonzero(np.abs(seg - g[b]) <= tol)
        start = a + int(still[still < settled[0]].max()) if (still < settled[0]).any() else a
        phases.append((start, a + int(settled[0]), change))
    return phases


def key_times(g, times, h=PHASE_MIN_CHANGE):
    """Data times of grasps (a closing has settled) and releases (an opening begins)."""
    times = np.asarray(times, float)
    return [float(times[b] if change < 0 else times[a]) for a, b, change in gripper_phases(g, times, h)]
