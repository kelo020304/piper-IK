#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${PIPER_IK_PYTHON:-}" ]]; then
  PYTHON_BIN="$PIPER_IK_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN="$PWD/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi
exec env -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" launch.py "$@"
