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
from typing import Mapping

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
    # Effective sample size (tsu_compiler.ess), measured on the SAME samples `got` in
    # search.py's `_verify` was already drawn from, kept unflattened per-chain
    # (thrml_backend.sample_chains) long enough to estimate an autocorrelation
    # time from. None with `ess_note` explaining why whenever the estimate is
    # not trustworthy (too few samples, or N/tau below tsu_compiler.ess's AR(1)-
    # validated reliability threshold) -- never a fabricated number for a
    # chain too short to support one.
    ess: float | None = None
    ess_note: str = "unavailable: not measured"
    # C4: diversity. `diversity_distinct` (distinct decoded, task-valid
    # configurations seen) and `diversity_valid_samples` (how many samples
    # that count is OVER) need no exact reference -- computed unconditionally
    # in `_verify`, same as task_validity/codeword_violation_rate, so they
    # default here only for a Verification built without running `_verify`
    # (e.g. an older test). `diversity_reachable` -- the SIZE of the whole
    # valid state space -- is the number that makes distinct/valid readable
    # rather than misleading on a small space; it needs the logical state
    # space to be enumerable and so is None-with-a-reason when it is not.
    diversity_distinct: int | None = None
    diversity_note: str = "unavailable: not measured"
    diversity_valid_samples: int | None = None
    diversity_reachable: int | None = None
    diversity_reachable_note: str = "unavailable: not measured"
    # C1 (external review): TV certifies that we SAMPLED the distribution we
    # COMPILED. It says nothing about directed behaviour. SPR N-026/N-027
    # measured a flat energy-based model reproducing a ratchet's stationary
    # distribution to TV ~1e-15 while its net current went to ZERO -- so a
    # workload declaring `conserve_over_edges` (one binary spin per edge, read
    # as the flow from u to v) can agree perfectly on TV and still transport
    # nothing. Set for exactly those workloads, empty otherwise: a disclosure
    # printed on every compile is one a reader learns to skip.
    transport_note: str = ""

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
            "diversity_distinct": f(self.diversity_distinct, self.diversity_note),
            "diversity_valid_samples": self.diversity_valid_samples,
            "diversity_reachable": f(self.diversity_reachable,
                                    self.diversity_reachable_note),
            "transport": self.transport_note or "not applicable: this workload declares no directed flow variable",
        }


@dataclass(frozen=True)
class DecodedSample:
    """G2: ONE concrete decoded sample from the compile's own verification
    run -- the WORKLOAD's own variable names and values (`Encoded.decode`'s
    output), never raw physical spins. Task validity (`Verification.
    task_validity`) is measured over the whole run; an application that
    renders a world needs a single concrete state, which is what this is.

    Preferred: a sample that is BOTH a valid codeword and passes the task
    contract, so the rendered example is a real solution
    (`is_codeword=True, task_valid=True, violations=()`). When no such
    sample was drawn in this run, this instead carries a FAILING sample --
    a valid codeword that the task contract rejected, with its own specific
    violations (`is_codeword=True, task_valid=False, violations=(...)`).
    When not even one codeword was drawn, `decoded` is None and `note`
    explains why -- never a fabricated stand-in for either case.

    `seed`/`clamp` are its provenance: the sampling run's own fixed seed,
    and the WORKLOAD-level clamp (if any) that was active while it was
    drawn."""
    decoded: Mapping[str, int] | None
    is_codeword: bool
    task_valid: bool
    violations: tuple[str, ...]
    seed: int
    clamp: Mapping[str, int]
    note: str = ""

    def to_dict(self):
        return {
            "decoded": dict(self.decoded) if self.decoded is not None else None,
            "is_codeword": self.is_codeword,
            "task_valid": self.task_valid,
            "violations": list(self.violations),
            "seed": self.seed,
            "clamp": dict(self.clamp),
            "note": self.note,
        }


def tv_noise_floor(probs: np.ndarray, n: int) -> float:
    """A TV number without its finite-sample floor beside it is uninterpretable."""
    return float(0.5 * np.sum(np.sqrt(2.0 * probs * (1 - probs) / (math.pi * n))))
