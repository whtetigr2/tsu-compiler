"""Task 5 & 6: pure, headlessly-tested statistics behind the LATTICE demo's
REGIME (temperature) and SCOPE panels. No Tk here -- see
tests/test_scope.py for the headless suite; demo/lattice_app.py imports
these functions and does the Tk drawing.

TASK 5 -- beta_to_temperature: beta is inverse temperature. The
convention exists because exp(-beta*E) is tidier to carry through the
compiler than exp(-E/T), not because temperature is unusable -- both
belong on screen (see lattice_app.py's REGIME panel). beta<=0 is refused:
it is not a colder or hotter model, it is a meaningless one -- exp(-beta*E)
would invert (beta<0, low-energy states become LEAST likely) or flatten
(beta==0, every state equally likely, T undefined/infinite) the
distribution the rest of this app treats as physical.
"""
from __future__ import annotations


def beta_to_temperature(beta: float) -> float:
    """T = 1/beta. beta<=0 raises -- see module docstring."""
    if beta <= 0:
        raise ValueError(
            f"beta must be positive (got {beta!r}); beta<=0 is not a colder "
            f"or hotter model, it is a meaningless one -- exp(-beta*E) would "
            f"invert (beta<0) or flatten (beta==0) the distribution")
    return 1.0 / beta
