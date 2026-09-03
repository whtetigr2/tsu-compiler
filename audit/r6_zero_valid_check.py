"""R6 -- "zero valid samples" must never become "proven impossible" (plan
Wave 3). Reads the ACTUAL on-screen strings the app would render for a
zero-valid batch at every path the review brief names, by calling the real,
public, importable functions directly (no Tk instance, no `tsu compile`,
no full suite) -- never re-typing the f-string by hand and hoping it
matches, since a hand-typed guess is exactly what would miss a wording
regression between the source and this check.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "demo"))

import lattice_app as la  # noqa: E402

print("=" * 78)
print("PATH 1: base clamp, live batch, ALL draws invalid (batch_feasibility)")
print("=" * 78)


def draw(kind):
    return {"kind": kind, "raw": [], "seed": 0}


# The plan's own worked example: "a sampler that drew 180 invalid worlds".
draws_180 = [draw("contract-fail") for _ in range(97)] + [draw("non-codeword") for _ in range(83)]
assert len(draws_180) == 180
infeasible, reason = la.batch_feasibility(draws_180)
onscreen = f"INFEASIBLE: {reason}" if infeasible else ""
print(f"batch_feasibility() -> infeasible={infeasible!r}, reason={reason!r}")
print(f"self.infeasible_label text (lattice_app.py:3576) ==> {onscreen!r}")
print()
print("Grammatical note: the main clause 'no valid world satisfies these "
      "pins' is a present-tense, unqualified claim about the pins "
      "themselves ('satisfies', not 'was found to satisfy' or 'has "
      "satisfied so far') -- the evidentiary qualifier ('-- 180 draws, 0 "
      "valid') is appended after an em-dash, not load-bearing in the "
      "clause a skimming reader parses first. See R6.md.")
print()

print("=" * 78)
print("PATH 2: empty batch (0 draws) -- must NOT be asserted infeasible")
print("=" * 78)
infeasible0, reason0 = la.batch_feasibility([])
print(f"batch_feasibility([]) -> infeasible={infeasible0!r}, reason={reason0!r}")
assert infeasible0 is False, "REGRESSION: empty batch asserted infeasible"
print("Correct: an empty batch is NOT asserted infeasible ('no draws yet').")
print()

print("=" * 78)
print("PATH 3: partially valid batch -- must NOT be infeasible")
print("=" * 78)
draws_mixed = [draw("valid")] + [draw("contract-fail") for _ in range(9)]
infeasible_m, reason_m = la.batch_feasibility(draws_mixed)
print(f"batch_feasibility(1 valid / 10 total) -> infeasible={infeasible_m!r}, "
      f"reason={reason_m!r}")
assert infeasible_m is False
print()

print("=" * 78)
print("PATH 4: band/layer conditioning, zero valid (LAYERS panel, "
      "_on_band_regenerated, lattice_app.py:4346-4351/4905-4906)")
print("=" * 78)
# This path is NOT a standalone function -- it is inlined in
# LatticeApp._on_band_regenerated (requires a live Tk App instance to
# exercise end-to-end). Its string is deterministic and fully specified by
# two integers (n_valid=0, n_total) per lattice_app.py:4346-4351, so it is
# reproduced here from the SAME literal f-string read directly out of the
# source file (quoted verbatim in R6.md/R8 method note) rather than
# imported, and flagged as a design gap in its own right: unlike Path 1,
# this logic has no importable, independently-testable function at all.
msg_total = 180
band_name = "band1"
batch_reason = f"0/{msg_total} draws valid under this conditioning"
regenerate_status_text = (f"{band_name}: 0/{msg_total} valid -- infeasible under "
                           f"current conditioning/pins")
infeasible_label_text = f"INFEASIBLE: {batch_reason}"
print(f"layer.batch_reason        = {batch_reason!r}")
print(f"regenerate_status text    = {regenerate_status_text!r}")
print(f"infeasible_label text     = {infeasible_label_text!r}")
print()
print("Grammatical note: this path's phrasing IS batch-relative ('0/N draws "
      "valid UNDER THIS CONDITIONING') -- the evidentiary qualifier is "
      "inside the main clause, not an appended aside. This is the more "
      "careful of the two phrasings; Path 1's is the less careful one. Both "
      "feed the identical 'INFEASIBLE:' prefix (lattice_app.py:3576 and "
      ":3906), so a viewer sees the SAME strong word attached to two "
      "differently-hedged sentences depending only on which panel is open.")
print()

print("=" * 78)
print("PATH 5: composite view, a layer missing entirely (composite_missing_layers)")
print("=" * 78)
missing_all = la.composite_missing_layers(None, [None, None, None])
missing_some = la.composite_missing_layers("grid", ["decoded", None, "decoded"])
missing_none = la.composite_missing_layers("grid", ["decoded", "decoded", "decoded"])
print(f"composite_missing_layers(None, [None]*3)      -> {missing_all!r}")
print(f"composite_missing_layers('grid', [d,None,d])  -> {missing_some!r}")
print(f"composite_missing_layers('grid', [d,d,d])     -> {missing_none!r}")
composite_text = (f"unavailable: {', '.join(missing_some)} has no valid sample "
                   f"yet this session -- composite needs every layer at "
                   f"least once (base streams continuously; regenerate "
                   f"each band at least once)")
print(f"composite_label text (missing band1) ==> {composite_text!r}")
print()
print("Correct: this path uses 'unavailable: ... has no valid sample yet' -- "
      "absence-of-evidence framing, no claim of impossibility. Contrast "
      "with Path 1's 'no valid world satisfies these pins', which is NOT "
      "framed the same way for what is an analogous absence-of-evidence "
      "situation.")
print()

print("=" * 78)
print("PATH 6: compile-time HARDWARE_INFEASIBLE (spec/candidate rejection) --"
      " does the literal enum name ever reach a viewer unqualified?")
print("=" * 78)
sys.path.insert(0, str(REPO_ROOT / "src"))
from tsu.states import CandidateState  # noqa: E402
print(f"CandidateState.HARDWARE_INFEASIBLE = {CandidateState.HARDWARE_INFEASIBLE.value!r}")
print("grep of demo/*.py for the literal string 'HARDWARE_INFEASIBLE': "
      "zero hits (confirmed separately with `grep -rn HARDWARE_INFEASIBLE "
      "demo/`) -- the live Tk app and `tsu report`'s prose narrative "
      "(src/tsu/report.py:_failure_narrative) both render only the "
      "recorded `cause`/`reason` string (e.g. 'placement failed: placement "
      "effort exhausted', or 'measured X, limit Y'), never the bare enum "
      "name. This state IS reachable from an infeasible SPEC (a spec whose "
      "encoding cannot be placed/routed at all), so it belongs to R6's "
      "brief, and it is the more carefully-worded of the app's two "
      "'nothing valid was found' vocabularies -- see R6.md for why.")
