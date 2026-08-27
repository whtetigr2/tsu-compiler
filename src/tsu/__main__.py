"""Allow `python -m tsu ...` as well as the installed `tsu` command."""
import sys

from .cli import main

sys.exit(main())
