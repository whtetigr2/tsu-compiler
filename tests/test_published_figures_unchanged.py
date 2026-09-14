"""Do the shipped workloads still match the figures this project cites?

Every claim in `out/extropic-verify/` and in the write-ups rests on two things
staying true: that the codon models on disk are the ones whose numbers we quote,
and that each hardware limit is still classified the way its provenance says.

Both were checked by hand. A check run by hand is a check run when someone
remembers, which is why this project's own provenance audit made the exact-oracle
cross-check a test rather than a ritual. Same reasoning here.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.preflight.model import load_model
from tsu_compiler.target import PROFILES

PACK = Path(__file__).resolve().parents[1] / "out" / "extropic-verify"

# Extropic's own published figures for these workloads. Changing a number here
# is changing a claim about someone else's paper, which should never be a
# silent edit.
PUBLISHED = {
    "codon_spike_full": {"spins": 3147, "degree": 12},
    "codon_spike_200aa": {"spins": 481, "degree": 12},
    "codon_default_prefix": {"spins": 266, "degree": 12},
    "codon_tiny_10aa": {"spins": 31, "degree": 12},
}


@pytest.mark.parametrize("stem", sorted(PUBLISHED))
def test_the_shipped_codon_model_matches_the_figures_we_quote(stem):
    """WHAT THIS PINS: the model on disk is the one whose spin count and degree
    this project quotes in `EXTROPIC_VERIFICATION.md`, the write-ups, and the
    Extropic-facing pack. The full spike at 3,147 spins and degree 12 is the
    headline credential -- it is what makes "we compiled their own published
    workload" checkable rather than asserted.

    HOW IT FAILS: regenerate any `*.edges.json` with a different sequence length
    or penalty structure and the spin count or degree moves, while every document
    quoting it keeps the old number. Nothing else in the suite reads these files
    against their claimed figures.

    PROVENANCE: Extropic's codon_opt paper reports 3,147 spins at degree 12 for
    the SARS-CoV-2 spike; the smaller sizes are this project's own truncations of
    the same model, measured when the pack was built."""
    want = PUBLISHED[stem]
    model = load_model(edges=PACK / f"{stem}.edges.json")
    rep = analyse(model)
    assert rep.n_nodes == want["spins"], (
        f"{stem}: {rep.n_nodes} spins on disk, {want['spins']} quoted in the pack")
    assert rep.max_degree == want["degree"], (
        f"{stem}: degree {rep.max_degree} on disk, {want['degree']} quoted")


@pytest.mark.parametrize("stem", sorted(PUBLISHED))
def test_the_summary_agrees_with_its_own_edge_list(stem):
    """WHAT THIS PINS: each `*.summary.json` describes the `*.edges.json` beside
    it. A reader who trusts the summary without opening the edge list should not
    be misled.

    HOW IT FAILS: regenerate one file and not the other -- easy to do, since they
    are produced by separate steps -- and the summary silently describes a model
    that is no longer there.

    PROVENANCE: the two files are companions produced by the same verification
    run; agreement is the property that makes either quotable."""
    model = load_model(edges=PACK / f"{stem}.edges.json")
    rep = analyse(model)
    summary = json.loads((PACK / f"{stem}.summary.json").read_text(encoding="utf-8"))
    assert summary["n_spins"] == rep.n_nodes
    assert summary["max_degree"] == rep.max_degree
    assert summary["n_edges"] == rep.n_edges


@pytest.mark.parametrize("field,assumed", [
    ("max_abs_coupling", False),   # Thermalizers Fig. 12 cap-sweep axis
    ("node_budget", False),        # From One to One Billion, Fig. 05
    ("coupling_parameters", False),  # Z1T die callout
    ("degree", False),             # F-14
    ("max_abs_bias", True),        # this project's working value
    ("per_edge_independent_J", True),  # idealisation: ~9.89 edges per parameter
    ("connected_fabric", True),    # derived from the coupler count, not stated
])
def test_each_hardware_limit_keeps_its_provenance(field, assumed):
    """WHAT THIS PINS: whether each Z1 limit is a documented Extropic figure or
    this project's own assumption. That distinction is the tool's central claim
    -- a verdict resting only on an assumed limit says so and offers
    `--allow-assumed` -- and it is asserted here per field so a flag cannot be
    flipped quietly in either direction.

    HOW IT FAILS: promoting `max_abs_bias` to sourced without a citation would
    make the pack's one honest caveat disappear, and every report gating on |b|
    would start claiming hardware authority it does not have. Demoting
    `max_abs_coupling` to assumed would understate a figure Extropic published,
    which is the mirror error and one this project's README committed once
    already.

    PROVENANCE: `src/tsu_compiler/target.py`, where every field carries its own
    source string; this test asserts the classification, not the value."""
    assert PROFILES["z1"].is_assumed(field) is assumed, (
        f"{field} provenance changed: expected "
        f"{'assumed' if assumed else 'sourced'}")
