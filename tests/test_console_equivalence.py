"""Same IK and same commands as the data_viewer real-arm console (Pinocchio piper-IK), on the console's own
fitted drive model. The console runs in a subprocess with its own interpreter (console_reference.py); this test
process needs only numpy / scipy. Skipped where the console or the FastUMI dataset is not on the machine.

    FASTUMI_REPO=/path/to/fast-umi-100k FASTUMI_VIEWER_PYTHON=/path/to/console/python python -m pytest -q tests
"""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from piper_ik import PiperIK
from piper_ik.trajectory import Trajectory

REPO = Path(os.environ.get('FASTUMI_REPO', Path(__file__).resolve().parents[2]))
PY = os.environ.get('FASTUMI_VIEWER_PYTHON', '')
HAVE = bool(PY) and (REPO / 'data_viewer/runtime.py').is_file() and (REPO / 'FastUMI_100k_lerobot').is_dir() and Path(PY).is_file()
pytestmark = pytest.mark.skipif(not HAVE, reason='data_viewer console / FastUMI dataset / console interpreter not on this machine')


@pytest.fixture(scope='module')
def ref(tmp_path_factory):
    """Run the console once (about 20 s). Its environment fix is the one in data_viewer/run.sh: ROS noetic's
    libboost_python otherwise breaks Pinocchio."""
    out = tmp_path_factory.mktemp('console') / 'reference.npz'
    q = lambda code: subprocess.run([PY, '-c', code], capture_output=True, text=True).stdout.strip()
    prefix, site = q('import sys; print(sys.prefix)'), q('import sysconfig; print(sysconfig.get_path("purelib"))')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env.update(LD_LIBRARY_PATH=f'{site}/cmeel.prefix/lib:{prefix}/lib', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if Path(prefix, 'lib/libstdc++.so.6').is_file():
        env['LD_PRELOAD'] = str(Path(prefix, 'lib/libstdc++.so.6'))
    r = subprocess.run([PY, str(Path(__file__).with_name('console_reference.py')), str(REPO), str(out)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return np.load(out)


def test_same_ik_solutions_as_pinocchio_piper_ik(ref):
    """Same Pinocchio, same solver code, same model numbers and TCP: bit-identical joints."""
    ik = PiperIK(); dq = np.array([np.abs(ik.solve(T, s).q - q).max() for T, s, q in zip(ref['ik_targets'], ref['ik_seeds'], ref['ik_q'])])
    print(f'{len(dq)} solves (1/4 unreachable): |dq| max {dq.max():.1e} rad')
    assert (dq == 0).all()


def replay(ref, follow, own_mapping):
    ik = PiperIK(follow=follow)
    ik.start(ref[f'{follow}_m0'], .07, q_command=ref[f'{follow}_start_q'], anchor=ref[f'{follow}_anchor'])
    ik.load_trajectory(ref[f'{follow}_rows'], ref[f'{follow}_times'], keys=ref[f'{follow}_keys'])
    assert np.abs(ik._traj.poses - ref[f'{follow}_poses']).max() < 1e-12                       # same targets
    if not own_mapping:
        ik._traj = Trajectory(ref[f'{follow}_times'], ref[f'{follow}_poses'], ref[f'{follow}_rows'][:, 6])
    worst = {'q_rad': 0., 'time_s': 0., 'gripper_m': 0.}
    for row in ref[f'{follow}_log']:
        q, w = ik.tick(row[:6], row[6])
        worst['q_rad'] = max(worst['q_rad'], float(np.abs(q - row[7:13]).max()))
        worst['time_s'] = max(worst['time_s'], abs(ik.source_time - row[13])); worst['gripper_m'] = max(worst['gripper_m'], abs(w - row[14]))
    print(follow, 'own mapping' if own_mapping else 'console targets', len(ref[f'{follow}_log']), 'cycles', worst)
    return ik, worst


def test_strict_replay_same_commands(ref):
    """1x data time with this package's own FastUMI mapping: its targets agree to ~1e-16, commands to float noise."""
    ik, worst = replay(ref, 'strict', True)
    assert ik.finished and worst['q_rad'] < 1e-12 and worst['time_s'] == 0 and worst['gripper_m'] == 0


def test_wait_replay_same_commands(ref):
    """追点 with the console's targets: bit-identical commands, data time and gripper over the whole replay. (With
    this package's own mapping the ~1e-16 target noise reaches the pacing thresholds after a few hundred cycles and
    the timing drifts by under 1 ms, so that case is not compared.)"""
    ik, worst = replay(ref, 'wait', False)
    assert ik.finished and worst['q_rad'] == 0 and worst['time_s'] == 0 and worst['gripper_m'] == 0
