"""Operating-regime analysis. Cheap fields only in the vertical slice.

WL5/WL6 established that range, resolution and connectivity interact and that the
feasible region is an INTERIOR ISLAND. A compile reporting only "gates passed" has
not said whether the program sits anywhere usable. Fields needing sweeps read None
and the regime reads "unmeasured" -- never a guess.
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

    def to_dict(self):
        return {
            "energy_scale": self.energy_scale,
            "coupling_utilisation": self.coupling_utilisation,
            "precision_headroom": self.precision_headroom,
            "beta_recommendation": self.beta_recommendation,
            "mixing_indicator": self.mixing_indicator
                if self.mixing_indicator is not None else "unmeasured",
            "regime": self.regime,
            "basis": self.basis,
        }


def analyse_regime(report, target, energy_scale=None) -> RegimeReport:
    cap = target.max_abs_coupling.value
    # The cap applies to |J| AND |b| alike (spec section 7.1); reporting J alone
    # understated utilisation for any model whose bias, not its coupling, was the
    # thing actually close to (or over) the cap (C2).
    peak = max(report.max_abs_J, report.max_abs_b)
    util = None if cap == float("inf") else peak / cap

    bits = target.coupling_bits.value
    headroom = None
    if bits != float("inf") and report.max_abs_J > 0:
        step = (2 * report.max_abs_J) / (2 ** int(bits))
        headroom = report.max_abs_J / step if step > 0 else None

    if util is None:
        regime, basis = "unmeasured", "ideal target has no coupling cap"
    elif util > 1.0:
        regime, basis = "precision_limited", "coupling_utilisation > 1"
    elif headroom is not None and headroom < 4.0:
        regime, basis = "precision_limited", "fewer than 4 quantisation steps"
    else:
        regime, basis = "feasible", "within cap; mixing not measured"

    return RegimeReport(energy_scale, util, headroom, None, None, regime, basis)
