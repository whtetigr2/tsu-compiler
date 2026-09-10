"""Hardware targets are DATA, and every constraint carries its evidence.

A target profile that silently presents an assumption as a fact is the failure the
`source` field exists to prevent (spec section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

RULES = ((1, 0), (2, 1), (2, 3), (4, 1))


def _z1_offsets() -> tuple[tuple[int, int], ...]:
    """Rotations only. Including reflections gives degree 28 and breaks bipartiteness."""
    off = set()
    for a, b in RULES:
        off |= {(a, b), (-b, a), (-a, -b), (b, -a)}
    return tuple(sorted(off))


@dataclass(frozen=True)
class Sourced:
    value: Any
    source: str          # a fact id like "F-14", or "assumed", or "user-supplied"
    quote: str = ""


@dataclass(frozen=True)
class TargetProfile:
    """P-3/F-A5 + I-9a/F-R7: `max_abs_coupling` (the |J| cap) and
    `max_abs_bias` (the |b| cap) are TWO separate Sourced fields, even
    though both currently hold the identical numeric value 6.0. Before
    this split there was exactly ONE field feeding both `gates.py`'s
    `coupling_cap` and `field_cap` checks -- which meant a documented
    Extropic fact (|J| <= 6.0, Thermalizers 2608.01615v1.pdf Fig. 12's
    cap-sweep axis annotates the value 6 as "(Z1)") and a genuine,
    unsourced project assumption (|b| <= 6.0 -- h_max is named
    symbolically in the same paper but its numeric value is never stated)
    were disclaimed identically everywhere in the app. Never conflate
    these two fields again just because their VALUES happen to agree."""
    name: str
    degree: Sourced
    offsets: Sourced
    bipartite: Sourced
    schedule: Sourced
    max_abs_coupling: Sourced
    max_abs_bias: Sourced
    coupling_bits: Sourced
    node_budget: Sourced

    def is_assumed(self, field_name: str) -> bool:
        got = getattr(self, field_name)
        if not isinstance(got, Sourced):
            raise TypeError(f"{field_name} is not a Sourced field")
        return got.source == "assumed"

    def sourced_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self)
                     if isinstance(getattr(self, f.name), Sourced))


Z1 = TargetProfile(
    name="z1",
    degree=Sourced(16, "F-14",
                   "All nodes in Z1 have connection rules (1,0),(2,1),(2,3),(4,1), "
                   "giving degree 16"),
    offsets=Sourced(_z1_offsets(), "F-14/F-17"),
    bipartite=Sourced(True, "F-18", "Since the graph is 2-colorable"),
    schedule=Sourced("chromatic_block_gibbs", "F-19",
                     "resampling every node v in V1 ... The same is then done for V2"),
    # P-3/F-A5 + I-9a/F-R7: reclassified from "assumed" -- Jmax=6 is
    # Extropic's own documented Z1 coupling cap, not a project guess. Value
    # UNCHANGED (still 6.0); only the provenance is corrected.
    max_abs_coupling=Sourced(
        6.0, "2608.01615v1.pdf (Thermalizers) p.22, Fig. 12 cap-sweep axis: '6 (Z1)'",
        "cap-sweep axis values '0.3 0.5 0.75 1 1.5 2 3 6 (Z1) 10' annotate 6 as "
        "'(Z1)' -- Extropic's own Z1 hardware |J| cap; corroborated in prose "
        "(p.13: 'the hardware caps |J| <= Jmax, |h| <= hmax'; p.20: 'every "
        "programmable weight ... has finite dynamic range, |Jij| <= Jmax and "
        "|hi| <= hmax')"),
    # |b|'s cap (hmax) stays a genuine project assumption: hmax is named
    # SYMBOLICALLY in both prose quotes above, but no numeric value for it
    # appears anywhere in either primary source (grepped both extractions
    # for "hmax", "h_max", "6-bit", "bit-width", "quantiz*", "DAC" -- no
    # resolution). Do not merge this back into max_abs_coupling just
    # because the two currently share a value.
    max_abs_bias=Sourced(6.0, "assumed",
                         "project working value; NOT a sourced Extropic figure"),
    coupling_bits=Sourced(6, "assumed",
                          "project working value; NOT a sourced Extropic figure"),
    # A-2 (external review, 2026-09-09): reclassified from the Thermalizers
    # paper's approximate ~250,000-node estimate to the taped-out Z1 die's
    # own exact spec. `thermalizers` p.5 says, verbatim, "the entire chip
    # has ~250,000 nodes" -- a rounded figure used ONLY to size a per-
    # iteration Gibbs-sampling energy-cost calculation in that paragraph,
    # never presented as an architectural spec. `billion` ("From One to One
    # Billion: Torx, Thermalizers, and Z1" - Extropic.pdf) p.7, Fig. 05's
    # callout panel AND caption both independently state the die's exact
    # pbit count as a hardware spec -- caption: "The Z1 die: eight cores,
    # 269,568 pbits, 215,904 coupling parameters, >50 MHz sampling rate,
    # <1 W"; callout panel: "PBITS 269,568 IN 8 CORES". Both the panel and
    # the caption agree on 269,568, unambiguously, so this supersedes the
    # older approximate figure for the SAME quantity (audit/provenance.md
    # Part 1 row 9, and Finding P-1's now-resolved cross-document
    # discrepancy). A reader who encounters the retired "~250,000"/"F-15"
    # figure elsewhere in this project's history (e.g. committed receipts
    # predating this change) should treat 269,568 as current.
    #
    # This is also the SAFE direction to be wrong in only if the number is
    # right: a LARGER budget can produce a false PASS (a model that does
    # not actually fit reads as fitting), unlike the smaller, conservative
    # 250,000 which could only ever produce a false FAIL. The citation
    # above is unambiguous (the same exact figure in two places in the same
    # primary source, independently corroborated by this project's own
    # provenance audit), which is what makes accepting that direction of
    # risk appropriate here.
    #
    # Deliberately UNCHANGED by this fix: the SAME Fig. 05 gives two
    # DIFFERENT, unreconciled coupling-count figures for the die (a callout
    # panel's "2,135,904 COUPLERS" vs the caption's "215,904 coupling
    # parameters", roughly a 9.9:1 ratio) -- audit/findings/R2.md parks that
    # question explicitly as UNRESOLVED, since if the physical die really
    # shares couplers ~10:1 this project's one-independent-J-per-edge
    # modelling assumption would be wrong for physical Z1. That question is
    # untouched here; this field is about node/pbit COUNT only.
    node_budget=Sourced(
        269_568,
        "From One to One Billion: Torx, Thermalizers, and Z1 - Extropic.pdf "
        "(billion) p.7, Fig. 05 caption and callout panel",
        "caption: 'The Z1 die: eight cores, 269,568 pbits, 215,904 coupling "
        "parameters, >50 MHz sampling rate, <1 W'; callout panel: 'PBITS "
        "269,568 IN 8 CORES' -- Extropic's own exact architectural spec for "
        "the taped-out Z1 die. SUPERSEDES the earlier approximate figure "
        "sourced to a different Extropic document (thermalizers p.5, "
        "verbatim: 'the entire chip has ~250,000 nodes', a rounded estimate "
        "used only to size a per-iteration energy-cost calculation, never "
        "presented as a hardware spec). See audit/provenance.md Part 1 row "
        "9 and Finding P-1 for the full cross-document reconciliation. "
        "Coupling COUNTS (couplers vs coupling parameters) from this same "
        "figure remain unreconciled -- see audit/findings/R2.md -- and are "
        "unaffected by this change."),
)

IDEAL = TargetProfile(
    name="ideal",
    degree=Sourced(float("inf"), "control"),
    offsets=Sourced((), "control"),
    bipartite=Sourced(False, "control"),
    schedule=Sourced("chromatic_block_gibbs", "control"),
    max_abs_coupling=Sourced(float("inf"), "control"),
    max_abs_bias=Sourced(float("inf"), "control"),
    coupling_bits=Sourced(float("inf"), "control"),
    node_budget=Sourced(float("inf"), "control"),
)

PROFILES = {"z1": Z1, "ideal": IDEAL}
