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
    # `energy_scale` is the energy gap between the best PHYSICAL state that
    # decodes to a task-contract-satisfying answer and the best one that does
    # not (search.py's `_energy_scale`, patched in once `_verify` has the
    # encoded model and the spec's own contract in hand -- `analyse_regime`
    # itself has neither). None whenever it was never measured (the model
    # exceeds the exact-enumeration limit, or every state fell on the same
    # side of the contract so there is no gap to report) -- `energy_scale_note`
    # carries the reason, same convention as `precision_headroom_note` and
    # `mixing_indicator_note` above. Defaults to "" so a RegimeReport built
    # before energy_scale was ever attempted (every candidate's regime is
    # computed in `_try`, before `_verify` has run) still renders honestly.
    energy_scale_note: str = ""

    def to_dict(self):
        return {
            "energy_scale": self.energy_scale
                if self.energy_scale is not None else (self.energy_scale_note or "unmeasured"),
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


# I2: beta_recommendation is a WINDOW, not a point estimate, because this
# project has measured beta to have an INTERIOR optimum (module docstring:
# "the feasible region is an INTERIOR ISLAND") -- too low is noise, too high
# freezes the chain before it reaches the answer. Both ends of the window
# below are read off the SAME quantity -- the Boltzmann factor
# exp(-beta * energy_scale), the relative weight the stationary distribution
# puts on the best task-CONTRACT-VIOLATING state next to the best
# task-satisfying one -- at its two qualitatively different crossover points:
#
#   beta * energy_scale ~= 1   -> exp(-1)  ~= 0.37   (LO)
#   beta * energy_scale ~= 10  -> exp(-10) ~= 4.5e-5 (HI)
#
# Below LO the violating state is still a substantial fraction of the
# stationary mass -- the chain has not yet separated the answer from the
# noise. Above HI the violating state is suppressed by four orders of
# magnitude, which is the point of running the sampler at all, but block-
# Gibbs moves that must cross an energy barrier of ORDER energy_scale to
# escape a false minimum see their own acceptance rate fall as the SAME
# exp(-beta * barrier) factor -- so pushing beta further past HI starts
# trading correctness-at-stationarity for the chain's ability to ever REACH
# stationarity, exactly the freezing failure this project has measured.
# [1, 10] is therefore not a fitted constant, and not a claim that this IS
# the optimum -- it is the defensible span between "still exploring" and
# "confidently decided but not yet frozen", stated as a decade on the one
# quantity (beta * energy_scale) that both failure modes are read from.
_BETA_WINDOW_LO_FACTOR = 1.0
_BETA_WINDOW_HI_FACTOR = 10.0


def beta_recommendation_from_energy_scale(energy_scale: float | None) -> tuple | None:
    """(lo, hi) as ABSOLUTE beta values -- `_BETA_WINDOW_*_FACTOR / energy_scale`,
    i.e. 'expressed in units of' energy_scale per RegimeReport's own field.
    None whenever energy_scale itself is None (never measured, or unmeasurable
    -- see search.py's `_energy_scale`) or non-positive (a zero or negative
    gap has no crossover scale to read a window off of): a recommendation
    derived from a number that was never honestly measured would itself be
    fabricated, so this refuses to invent one rather than silently reporting
    a plausible-looking window."""
    if energy_scale is None or energy_scale <= 0:
        return None
    return (_BETA_WINDOW_LO_FACTOR / energy_scale, _BETA_WINDOW_HI_FACTOR / energy_scale)


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

    return RegimeReport(energy_scale, util, headroom,
                        beta_recommendation_from_energy_scale(energy_scale),
                        None, regime, basis, precision_headroom_note=headroom_note)
