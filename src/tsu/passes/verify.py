"""Three-layer equivalence. Never fabricates: absent a reference, the field reads
"unavailable" with a reason (spec section 10).

I8 (final review): every field here except `task_validity` carried its own note
alongside it (`energy_note`, `execution_note`, `cross_check_note`); `task_validity`
alone fell back to a bare `"unavailable"` literal with no reason attached, and
`execution_noise_floor` fell back to a raw `None` with no note at all -- the
visualizer then rendered both as the literal string "None". Both now carry (or
reuse) a note the same way their siblings always have.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Verification:
    energy_tv: float | None
    energy_note: str
    task_validity: float | None
    execution_tv: float | None
    execution_note: str
    execution_noise_floor: float | None
    cross_check_tv: float | None
    cross_check_note: str
    # C5: decode() is a projection, not an inverse -- handed a non-monotone
    # (invalid) chain it still returns a legal-looking value with no flag. A
    # sample search.py's decoder cannot vouch for must not silently count toward
    # task_validity; codeword_violation_rate makes that failure mode visible
    # instead of folding it into (and inflating) task_validity.
    codeword_violation_rate: float | None = None
    codeword_violation_note: str = ""
    # I8: task_validity's own reason, in the same form as every sibling field.
    # Defaults to the old bare literal so a Verification constructed without
    # setting it explicitly (e.g. an older caller, or a test) still renders the
    # same "unavailable" it always did -- but a real caller should always set
    # this to something informative when task_validity is None.
    task_validity_note: str = "unavailable"
    # Effective sample size (tsu.ess), measured on the SAME samples `got` in
    # search.py's `_verify` was already drawn from, kept unflattened per-chain
    # (thrml_backend.sample_chains) long enough to estimate an autocorrelation
    # time from. None with `ess_note` explaining why whenever the estimate is
    # not trustworthy (too few samples, or N/tau below tsu.ess's AR(1)-
    # validated reliability threshold) -- never a fabricated number for a
    # chain too short to support one.
    ess: float | None = None
    ess_note: str = "unavailable: not measured"

    def to_dict(self):
        def f(v, note):
            return v if v is not None else note
        return {
            "energy_tv": f(self.energy_tv, self.energy_note),
            "task_validity": f(self.task_validity, self.task_validity_note),
            "execution_tv": f(self.execution_tv, self.execution_note),
            # execution_noise_floor is computed IN THE SAME BRANCH as
            # execution_tv and is None exactly when execution_tv is -- reusing
            # execution_note (rather than inventing a duplicate-text field) is
            # honest, not a shortcut.
            "execution_noise_floor": f(self.execution_noise_floor, self.execution_note),
            "cross_check_tv": f(self.cross_check_tv, self.cross_check_note),
            "codeword_violation_rate": f(self.codeword_violation_rate,
                                        self.codeword_violation_note),
            "ess": f(self.ess, self.ess_note),
        }


def tv_noise_floor(probs: np.ndarray, n: int) -> float:
    """A TV number without its finite-sample floor beside it is uninterpretable."""
    return float(0.5 * np.sum(np.sqrt(2.0 * probs * (1 - probs) / (math.pi * n))))
