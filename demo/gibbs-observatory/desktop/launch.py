"""Frozen entry script.

PyInstaller runs this file directly, not `python -m desktop.main`, so a frozen
script has no parent package and cannot use relative imports. The parent
directory goes on the path first so `desktop` is importable as a package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
