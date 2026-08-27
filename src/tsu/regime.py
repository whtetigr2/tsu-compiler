"""Operating-regime analysis. Cheap fields only in the vertical slice.

WL5/WL6 established that range, resolution and connectivity interact and that the
feasible region is an INTERIOR ISLAND. A compile reporting only "gates passed" has
not said whether the program sits anywhere usable. Fields needing sweeps read None
and the regime reads "unmeasured" -- never a guess.

I1 (final review): `precision_headroom` used to be computed as
`|J|max / ((2*|J|max) / 2**bits)`, which reduces algebraically to `2**(bits-1)` --
ALWAYS, independent of the model. It was a constant reported as a measurement (every
z1 receipt read 32.0), and it was not the spec's quantity anyway, which is
"quantisation step vs smallest distinct |J| gap": the real risk quantisation poses is
two DIFFERENT couplings landing on the same (or adjacent) quantised level and
becoming indistinguishable, not some ratio of the largest coupling to itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeReport:
    energy_scale: float | None
    coupling_utilisation: float | None
    precision_headroom: float | None
    beta_recommendation: tuple | None
    mixing_indicator: float | None
    regime: str
    basis: str
    precision_headroom_note: str = ""
    # mixing_indicator is the integrated autocorrelation time (tau) tsu.ess
    # estimated from the compile's own sampling run, when the chain supported
    # a reliable estimate (see search.py's compile_spec, which patches this
    # field in once verification has actually sampled). `mixing_indicator_note`
    # carries the reason when it did not (too few samples, or N/tau below
    # tsu.ess's reliability threshold) -- defaults to "" so a RegimeReport
    # built before mixing was ever attempted (every candidate's regime is
    # computed in `_try`, before any sampling has happened) still renders the
    # same bare "unmeasured" it always did.
    mixing_indicator_note: str = ""

    def to_dict(self):
        return {
            "energy_scale": self.energy_scale,
            "coupling_utilisation": self.coupling_utilisation,
            "precision_headroom": self.precision_headroom
                if self.precision_headroom is not None else self.precision_headroom_note,
            "beta_recommendation": self.beta_recommendation,
            "mixing_indicator": self.mixing_indicator
                if self.mixing_indicator is not None
                else (self.mixing_indicator_note or "unmeasured"),
            "regime": self.regime,
            "basis": self.basis,
        }


def _quantisation_step(target) -> float | None:
    """The quantisation step on the target's FIXED grid (the cap does not move
    with the model): 2 * max_abs_coupling / 2**coupling_bits. None when the
    target has no finite cap or bit width to quantise onto (e.g. IDEAL)."""
    bits, cap = target.coupling_bits.value, target.max_abs_coupling.value
    if bits == float("inf") or cap == float("inf"):
        return None
    return (2.0 * cap) / (2 ** int(bits))


def _smallest_distinct_gap(weights) -> float | None:
    """Smallest nonzero gap between distinct |J| magnitudes actually present in
    the model. None when there are fewer than two distinct couplings -- with
    zero or one distinct |J|, there is nothing for quantisation to collide."""
    if weights is None:
        return None
    mags = sorted({round(float(abs(w)), 12) for w in weights})
    if len(mags) < 2:
        return None
    return min(b - a for a, b in zip(mags, mags[1:]))


def analyse_regime(report, target, weights=None, energy_scale=None) -> RegimeReport:
    cap = target.max_abs_coupling.value
    # The cap applies to |J| AND |b| alike (spec section 7.1); reporting J alone
    # understated utilisation for any model whose bias, not its coupling, was the
    # thing actually close to (or over) the cap (C2).
    peak = max(report.max_abs_J, report.max_abs_b)
    util = None if cap == float("inf") else peak / cap

    step = _quantisation_step(target)
    gap = _smallest_distinct_gap(weights)
    headroom, headroom_note = None, ""
    if step is None:
        headroom_note = "unmeasured: target has no finite coupling cap/bit width"
    elif gap is None:
        headroom_note = "unmeasured: fewer than two distinct |J| values in the model"
    elif step > 0:
        headroom = gap / step

    if util is None:
        regime, basis = "unmeasured", "ideal target has no coupling cap"
    elif util > 1.0:
        regime, basis = "precision_limited", "coupling_utilisation > 1"
    elif headroom is not None and headroom < 1.0:
        regime, basis = ("precision_limited",
                         "two distinct |J| values are closer together than one "
                         "quantisation step and may become indistinguishable")
    else:
        regime, basis = "feasible", "within cap; mixing not measured"

    return RegimeReport(energy_scale, util, headroom, None, None, regime, basis,
                        precision_headroom_note=headroom_note)
