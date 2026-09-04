"""Thermometer elevation: N binary overlays counted give N+1 ordinal levels.

Each "band" is a decoded assignment of the SAME compiled overlay receipt
(demo/receipts/elev_band, see specs/lattice_elev_band_8x8.yaml) -- one
binary spin per cell, encouraged (never forced) toward 1 ("above this
band's threshold") where the terrain below supports it. Counting how many
bands are true at a cell gives that cell's elevation LEVEL: 0 if no band is
true, up to N if every band is true. This is a thermometer code -- ordinal
by construction -- and it is what recovers the "gradual transitions"
property that was lost when the base layer's k dropped from 5 to 3 (see
specs/lattice_small_8x8_k3.yaml's own docstring on the field-cap arithmetic
that forced that drop).

Cross-layer rules here are computed from DECODED layers at sample time and
folded into the next band's draw via demo/layers.bias_patch -- never baked
into the overlay spec itself, or the overlay would need recompiling per
base world, defeating the entire amortized-compile design (one receipt at
demo/receipts/elev_band serves every band of every world).

THE CATCH, measured not assumed: band i+1 should only be true where band i
is true (a proper thermometer code never has a "hole" -- 0110 is invalid,
only 0001/0011/0111/1111 etc. are). `band_patch` expresses that as an
ENCOURAGED bias nudge (weight > 0 at the band-below's true cells, weight <
0 at its false cells) -- not a hard constraint, so real draws CAN break
monotonicity. `monotonicity_violations` exists to COUNT that breakage, not
hide it: silently sorting bands into a valid thermometer order would
destroy the very violation rate a later task is required to measure (spec
section 3.1). See that function's own docstring.

FIELD_CAP (|b| <= 6.0) is an assumed project working value, not a sourced
Extropic figure -- see demo/layers.py and demo/receipts/small/target.json's
own "assumed" tag on max_abs_bias (F-R2: not max_abs_coupling, the |J| cap,
which is now Extropic-documented -- see P-3/F-A5 + I-9a/F-R7). `strength`
in `band_patch` is an
unscaled per-cell nudge; callers driving this through demo/layers.bias_patch
get FieldCapExceeded for free if a chosen strength pushes |b|max over cap.
"""
from __future__ import annotations

from typing import Mapping


def thermometer_level(bands: list[dict], cell: str) -> int:
    """Count how many of `bands` (each a decoded {cell_name: 0|1} assignment,
    e.g. what `enc.decode` returns for a compiled elev_band draw) are true
    (== 1) at `cell`. This is the thermometer READOUT, not a validity check
    -- it counts true bands regardless of whether they form a clean prefix
    (see `monotonicity_violations` for that separate concern)."""
    return sum(1 for band in bands if int(band[cell]) == 1)


def monotonicity_violations(bands: list[dict]) -> list[tuple[str, int]]:
    """Return every (cell, i) where band `i` is true but band `i-1` is
    false -- a broken rung in the thermometer, `i` being the HIGHER band's
    index (so `i` in [1, len(bands)-1]).

    DETECTS, NEVER REPAIRS. `band_patch`'s monotonicity nudge is an
    encouragement, not a constraint (see this module's docstring), so real
    sampled bands legitimately break it -- sorting, clamping, or otherwise
    repairing the bands here would silently launder that breakage instead
    of reporting the rate a later task (spec section 3.1) needs measured.
    """
    cells = bands[0].keys() if bands else ()
    out: list[tuple[str, int]] = []
    for i in range(1, len(bands)):
        lower, upper = bands[i - 1], bands[i]
        for cell in cells:
            if int(upper[cell]) == 1 and int(lower[cell]) == 0:
                out.append((cell, i))
    return out


def band_patch(prev_band: dict | None, base: dict,
               strength: float) -> dict[tuple[str, int], float]:
    """Build the {(cell, 1): weight} patch (see demo/layers.bias_patch's own
    docstring for the weight sign convention: weight > 0 ENCOURAGES that
    (cell, value) pair) for the NEXT band to sample, conditioned on the
    band immediately below it (`prev_band`, None for band 0) and on the
    base terrain layer (`base`, a decoded base-layer assignment: 0 water, 1
    rock, 2 grass per specs/lattice_small_8x8_k3.yaml).

    Band 0 (`prev_band is None`) conditions on the base layer alone: higher
    ground away from water is encouraged toward "above threshold" (1),
    water cells discouraged. Band i>0 conditions on band i-1's own decoded
    truth PLUS the base layer -- encouraged toward 1 where band i-1 is
    true (the monotonicity nudge this module's docstring describes),
    discouraged where band i-1 is false, modulated by the same base-layer
    term band 0 uses so every band still respects "not above threshold
    over open water".

    `strength` is an unscaled per-cell magnitude (positive); the caller
    multiplies it against each cell's condition to get that cell's signed
    weight, matching demo/stacked_world.py's own ALPHA-as-dose-response
    convention for this same bias_patch mechanism.
    """
    strength = float(strength)
    patch: dict[tuple[str, int], float] = {}
    for cell, base_value in base.items():
        # Base-layer term: water (0) discourages "above threshold", rock (1)
        # and grass (2) encourage it in proportion to how far from water
        # they sit on the WATER/ROCK/GRASS ordinal (0 lowest, 2/1 higher) --
        # deliberately simple (base_value - 1, so water=-1, rock=0, grass=1)
        # rather than importing render_world's distance-transform machinery,
        # which this function has no access to (it only sees the decoded
        # dict, not the grid geometry).
        base_term = float(int(base_value)) - 1.0

        if prev_band is None:
            weight = strength * base_term
        else:
            # Monotonicity is the DIRECTION-setting term (band i>0 must
            # never let terrain override "below is false" into a false
            # encourage, or the thermometer's own ordering guarantee -- the
            # one thing band_patch is supposed to bias toward -- would be
            # undermined by its own base-layer term). base_term only
            # MODULATES magnitude, via a strictly-positive factor (base
            # values are in {0,1,2} here, so base_term in {-1,0,1} and this
            # factor stays in [0.5, 1.5]) -- it can never flip the sign
            # below_term set.
            below_term = 1.0 if int(prev_band[cell]) == 1 else -1.0
            base_factor = 1.0 + 0.5 * base_term
            weight = strength * below_term * base_factor

        patch[(cell, 1)] = weight
    return patch
