"""Allow `python -m tsu_compiler ...` as well as the installed `tsu` command.

The guard matters: without it, merely IMPORTING this module runs the CLI, which
argparse then aborts for want of arguments. A guard test caught exactly that.
"""
import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
