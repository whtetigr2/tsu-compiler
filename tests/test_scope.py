"""Task 5 & 6: demo/scope.py's pure, headlessly-tested statistics -- the
temperature readout (Task 5) and the SCOPE panel's four readouts (Task 6:
autocorrelation, magnetization, energy histogram, local-field/sigmoid
response). Every test here checks against a signal with a KNOWN ANALYTIC
ANSWER, never against scope.py's own output -- see the module docstring
in demo/scope.py for why.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))


# ---------------------------------------------------------------------------
# Task 5: beta_to_temperature
# ---------------------------------------------------------------------------

def test_temperature_is_the_reciprocal_of_beta():
    from scope import beta_to_temperature
    assert beta_to_temperature(1.0) == 1.0
    assert beta_to_temperature(4.0) == 0.25


def test_zero_or_negative_beta_is_refused():
    """beta <= 0 is not a colder or hotter model, it is a meaningless one --
    exp(-beta*E) would invert or flatten the distribution."""
    from scope import beta_to_temperature
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            beta_to_temperature(bad)
