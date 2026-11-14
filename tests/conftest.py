import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]           # the package folder (the repository root on GitHub)
if PKG.name == 'piper_ik':
    if str(PKG.parent) not in sys.path:
        sys.path.insert(0, str(PKG.parent))          # the folder that contains piper_ik/
else:                                                # a clone under another name: needs `pip install -e .`
    try:
        import piper_ik  # noqa: F401
    except ImportError:
        raise SystemExit(f'{PKG} is not named piper_ik: clone it as `git clone <url> piper_ik`, or run `pip install -e {PKG}`')
