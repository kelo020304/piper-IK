"""Record what the data_viewer console does, for test_console_equivalence.py. Runs in the console's own environment
(its interpreter, Pinocchio, the original piper-IK), never in the test process:

    python console_reference.py <repo> <out.npz>

1. Pinocchio piper-IK solves of random targets / seeds with the hardware limits.
2. The console Runtime replaying Prepare_tableware ep247 on its fitted drive model (data_viewer tests), per cycle:
   measured joints, cycle time, command, data time, gripper command; for follow 'wait' and 'strict'.
"""
import sys
import tempfile
from pathlib import Path
import numpy as np

repo = Path(sys.argv[1]); out = Path(sys.argv[2]); viewer = repo / 'data_viewer'
sys.path[:0] = [str(viewer), str(viewer / 'tests')]
import test_runtime as T                                    # noqa: E402  the console's fitted drive model and EP247 settings
from runtime import Runtime, hardware_limits                 # noqa: E402
from runtime_motion import make_ik                           # noqa: E402
from data import Catalog                                     # noqa: E402
from robot import DEFAULT_XML, DEFAULT_Q, Robot              # noqa: E402

data = {}
ik, _ = make_ik(Robot(DEFAULT_XML, np.array(DEFAULT_Q)), 'pinocchio', np.array(DEFAULT_Q)); lo, hi = hardware_limits()
ik._model.lower[:] = lo; ik._model.upper[:] = hi
rng = np.random.default_rng(7); targets, seeds, sols = [], [], []
for i in range(300):
    q = rng.uniform(lo, hi); Tg = ik.fk(q)
    if i % 4 == 3:
        Tg[:3, 3] += rng.normal(size=3) * .3
    s = np.clip(q + rng.normal(size=6) * (.05 if i % 2 else .6), lo, hi)
    targets.append(Tg); seeds.append(s); sols.append(ik.solve(Tg, s).q)
data.update(ik_targets=np.array(targets), ik_seeds=np.array(seeds), ik_q=np.array(sols))
grip = {'enabled': True, 'zero_width_m': 0., 'one_width_m': .07, 'effort_raw': 1000}
for follow in ('wait', 'strict'):
    sdk = T.PlantSDK(1.); sdk.place(T.EP247['start_q_rad'])
    rt = Runtime(Catalog(repo / 'FastUMI_100k_lerobot'), DEFAULT_XML, Path(tempfile.mkdtemp()), True, 'ref', lambda *a, **k: sdk)
    rt.shutdown.set(); rt.worker.join()
    try:
        now = 1000.; rt.connect('c', True); rt.tick(now); rt.action('c', 'arm')
        assert rt.check_path('c', T.EP247)['start_reachable']
        start_q = np.array(T.EP247['start_q_rad']); rt._reset(start_q); m0 = rt.measured.copy()
        rt.start('c', {**T.EP247, 'replay': True, 'record': False, 'mapping_confirmed': True, 'follow': follow, 'gripper': grip})
        sch = rt.schedule; log = []
        while rt.phase == 'playing' and len(log) < 15000:
            now += .01; sdk.step(); rt.tick(now); s = rt.last_sample
            log.append(np.r_[s['measured_q_rad'], s['control_interval_s'], s['command_q_rad'], s['source_time_s'], s['command_gripper_width_m']])
        log = np.array(log)
        data.update({f'{follow}_m0': m0, f'{follow}_start_q': start_q, f'{follow}_anchor': rt.anchor, f'{follow}_rows': sch.actions, f'{follow}_times': sch.times,
                     f'{follow}_keys': np.array(sch.key_times), f'{follow}_poses': sch.poses, f'{follow}_log': log})
    finally:
        rt.close()
np.savez(out, **data)
print('wrote', out)
