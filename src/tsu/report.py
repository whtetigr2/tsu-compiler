"""The compilation report: a fixed-format text rendering of a receipt, and
NOTHING else. `render_report(receipt_dir)` reads only the JSON/text files a
receipt already contains -- it never recomputes a number the compiler did not
already record, and it never hardcodes a verdict as a string literal (that was
the exact defect viz.py's own docstring documents from an earlier prototype:
"Oracle verdict: GREEN" printed regardless of what the receipt actually said).

Any field the receipt does not contain prints `unavailable: <reason>` here --
never a blank, a zero, or an invented value. ESS, Mixing and Diversity are not
computed anywhere in this compiler today, so they always read
`unavailable: not measured`; that is a true statement about what was measured,
not a placeholder standing in for a number nobody has.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The noise-floor multiple SAMPLING is judged against, matching the same
# tolerance test_4 (spec section 12) uses for "sampled matches exact" -- not a
# new, independently invented threshold.
_SAMPLING_TV_MULTIPLE = 5.0
_SAMPLING_TV_FLOOR = 0.05

_KERNEL_LABELS = {"chromatic_block_gibbs": "chromatic block Gibbs"}


def _load(d: Path, name: str) -> dict:
    p = d / name
    return json.loads(p.read_text()) if p.exists() else {}


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _as_unavailable(text: str) -> str:
    """Every note in this codebase that stands in for a missing number is
    supposed to read `unavailable: <reason>` (spec section 10) -- most already
    do (energy_note, task_validity_note, ...); a few upstream notes (regime.py's
    precision_headroom_note, receipt.py's mediators_note) were written as bare
    reasons without that prefix. Normalise here rather than editing every
    producer, since the producers' own JSON is correct on its own terms."""
    return text if text.lower().startswith("unavailable") else f"unavailable: {text}"


def _metric(metrics: dict, verdict: str, key: str, note_key: str | None = None) -> Any:
    if key in metrics and metrics[key] is not None:
        return metrics[key]
    if note_key and metrics.get(note_key):
        return _as_unavailable(metrics[note_key])
    if verdict != "COMPILED":
        return _as_unavailable(
            f"representation was never reached (verdict={verdict})")
    return "unavailable: not recorded"


def _verif(verification: dict, key: str) -> Any:
    if key not in verification or verification[key] is None:
        return "unavailable: not recorded"
    v = verification[key]
    return _as_unavailable(v) if isinstance(v, str) else v


def _regime(regime: dict, key: str) -> Any:
    if key not in regime or regime[key] is None:
        return "unavailable: not recorded"
    v = regime[key]
    return _as_unavailable(v) if isinstance(v, str) else v


def _gate(gates: list, name: str):
    """Returns the GateCheck dict for `name`, or None if it was never
    evaluated (e.g. the compile never reached gate checking)."""
    return next((g for g in gates if g["gate"] == name), None)


def _representative_row(candidates: list):
    """The candidates.json row to source REPRESENTATION/Z1-MAPPING fallback
    values from when metrics.json is empty (no candidate was SELECTED): the
    SELECTED row if one exists (it never will when metrics.json is empty, but
    this keeps the two lookups consistent), else the first candidate
    (declaration order) that actually reached `analyse` -- e.g. a HARDWARE
    verdict's degree-exceeded candidate still measured logical spins/edges
    before the degree gate rejected it. None when no candidate ever did."""
    for row in candidates:
        if row.get("state") == "SELECTED":
            return row
    return next((row for row in candidates if row.get("logical_spins") is not None),
               None)


def _rep_raw(metrics: dict, row: dict | None, metrics_key: str, row_key: str):
    """The raw value (never a formatted string) for a REPRESENTATION/Z1-MAPPING
    field: prefer metrics.json (the SELECTED candidate's own numbers), else
    fall back to a representative rejected candidate's row from
    candidates.json. Returns (value, from_fallback); value is None when
    neither source has it."""
    if metrics_key in metrics and metrics[metrics_key] is not None:
        return metrics[metrics_key], False
    if row is not None and row.get(row_key) is not None:
        return row[row_key], True
    return None, False


def _rep_display(metrics: dict, row: dict | None, verdict: str, metrics_key: str,
                 row_key: str, note_key: str | None = None) -> Any:
    """Display form of `_rep_raw`: the raw value if it came from metrics.json
    (the selected candidate), the value tagged with its source if it came from
    a fallback row (not the winning representation), else unavailable with a
    reason."""
    value, from_fallback = _rep_raw(metrics, row, metrics_key, row_key)
    if value is not None:
        return (f"{_fmt(value)} (from {row['encoding']}, {row['state']}; "
               f"not selected)" if from_fallback else value)
    if note_key and metrics.get(note_key):
        return _as_unavailable(metrics[note_key])
    if verdict != "COMPILED":
        return _as_unavailable(
            f"representation was never reached (verdict={verdict})")
    return "unavailable: not recorded"


def _selected_candidate(passes: dict):
    return next((c for c in passes.get("candidates", ())
                if c["state"] == "SELECTED"), None)


def _rejected_candidate(passes: dict):
    """The first (declaration-order) candidate that was actually rejected --
    used to source the failure narrative when the compile did not COMPILE."""
    return next((c for c in passes.get("candidates", ())
                if c["state"] in ("SEMANTICALLY_INVALID", "HARDWARE_INFEASIBLE")),
               None)


def _line(label: str, value: Any, width: int = 24) -> str:
    return f"  {label:<{width}}{_fmt(value)}"


def render_report(receipt_dir) -> str:
    d = Path(receipt_dir)
    workload = _load(d, "workload.json")
    target = _load(d, "target.json")
    metrics = _load(d, "metrics.json")
    passes = _load(d, "passes.json")
    candidates_table = _load(d, "candidates.json") or []
    gates = _load(d, "gates.json")
    regime = _load(d, "regime.json")
    verification = _load(d, "verification.json")
    program = _load(d, "program.json")

    verdict = passes.get("verdict", "unavailable")
    target_name = target.get("name", "?")

    lines = ["TSU COMPILATION REPORT", "─" * 32, ""]

    # -- WORKLOAD -----------------------------------------------------------
    lines.append("WORKLOAD")
    lines.append(_line("Variables:", workload.get("variables", "unavailable: not recorded")))
    lines.append(_line("Logical interactions:",
                       workload.get("logical_interactions", "unavailable: not recorded")))
    lines.append(_line("State cardinality:",
                       workload.get("state_cardinality", "unavailable: not recorded")))
    lines.append(_line("Constraint classes:",
                       workload.get("constraint_classes", "unavailable: not recorded")))
    lines.append("")

    # -- REPRESENTATION -------------------------------------------------------
    # Prefer metrics.json (the SELECTED candidate's own numbers); fall back to
    # a rejected candidate's row from candidates.json when no candidate was
    # selected -- e.g. too_dense.yaml's degree-exceeded candidate still
    # reached `analyse` before Z1's degree gate rejected it, so "logical
    # variables/edges/bipartite" are NOT actually unmeasured for that compile,
    # only unselected. `max_degree` has no fallback: A5's compare() table does
    # not carry it (its own failure narrative already names it instead).
    selected = _selected_candidate(passes)
    rep_row = _representative_row(candidates_table)
    encoding_display = (selected["encoding"].replace("_", "-") if selected else
                        _as_unavailable(f"no candidate selected (verdict={verdict})"))

    n_nodes_display = _rep_display(metrics, rep_row, verdict, "n_nodes", "logical_spins")
    n_edges_display = _rep_display(metrics, rep_row, verdict, "n_edges", "logical_edges")
    bip_raw, bip_fallback = _rep_raw(metrics, rep_row, "bipartite", "bipartite")
    if bip_raw is None:
        graph_display = _metric(metrics, verdict, "bipartite")
    else:
        graph_display = "bipartite" if bip_raw else "non-bipartite"
        if bip_fallback:
            graph_display += (f" (from {rep_row['encoding']}, {rep_row['state']}; "
                             f"not selected)")

    lines.append("REPRESENTATION")
    lines.append(_line("Encoding:", encoding_display))
    lines.append(_line("Logical variables:", n_nodes_display))
    lines.append(_line("Logical edges:", n_edges_display))
    lines.append(_line("Graph:", graph_display))
    lines.append(_line("Max degree:", _metric(metrics, verdict, "max_degree")))
    lines.append("")

    # -- Z1 MAPPING -----------------------------------------------------------
    # Raw values ONLY here (never a pre-formatted display string) -- direct
    # couplings and physical p-bits are computed by arithmetic on them, and a
    # formatted "N (from ...)" string must never leak into that arithmetic.
    n_edges, n_edges_fb = _rep_raw(metrics, rep_row, "n_edges", "logical_edges")
    n_nodes, n_nodes_fb = _rep_raw(metrics, rep_row, "n_nodes", "logical_spins")
    mediators_raw, mediators_fb = _rep_raw(metrics, rep_row, "mediators", "mediators")
    mediators_known = isinstance(mediators_raw, int)
    fallback_tag = (f" (from {rep_row['encoding']}, {rep_row['state']}; not selected)"
                    if rep_row else "")

    if mediators_known and n_edges is not None:
        direct_couplings = f"{n_edges - mediators_raw:,}" + (
            fallback_tag if (mediators_fb or n_edges_fb) else "")
    else:
        direct_couplings = _rep_display(metrics, rep_row, verdict, "mediators",
                                        "mediators", "mediators_note")
    if mediators_known and n_nodes is not None:
        physical_pbits = f"{n_nodes + mediators_raw:,}" + (
            fallback_tag if (mediators_fb or n_nodes_fb) else "")
    else:
        physical_pbits = _rep_display(metrics, rep_row, verdict, "mediators",
                                      "mediators", "mediators_note")
    mediators = _rep_display(metrics, rep_row, verdict, "mediators", "mediators",
                             "mediators_note")

    def _gate_display(g):
        if g is None:
            return _as_unavailable(f"gate not evaluated (verdict={verdict})")
        if g["passed"]:
            return "PASS"
        if g["downgraded"]:
            return "FAIL (downgraded via --allow-assumed)"
        return "FAIL"

    coupling_range = _gate_display(_gate(gates, "coupling_cap"))
    field_range = _gate_display(_gate(gates, "field_cap"))
    headroom = _regime(regime, "precision_headroom")
    coupling_precision = (headroom if isinstance(headroom, str) else
                          "PASS" if headroom >= 1.0 else "FAIL")

    lines.append("Z1 MAPPING")
    lines.append(_line("Direct couplings:", direct_couplings))
    lines.append(_line("Mediators required:", mediators))
    lines.append(_line("Physical p-bits:", physical_pbits))
    lines.append(_line("Coupling range:", coupling_range))
    lines.append(_line("Coupling precision:", coupling_precision))
    lines.append(_line("Field range:", field_range))
    lines.append("")

    # -- SAMPLING ---------------------------------------------------------
    schedule = program.get("schedule")
    kernel = _KERNEL_LABELS.get(schedule, schedule) if schedule else \
        _as_unavailable(f"no program built (verdict={verdict})")

    lines.append("SAMPLING")
    lines.append(_line("Kernel:", kernel))
    lines.append(_line("Colour blocks:",
                       _rep_display(metrics, rep_row, verdict, "colour_blocks",
                                    "colour_blocks")))
    # ESS/Mixing/Diversity are not computed anywhere in this compiler today --
    # print the honest state, not a placeholder pretending toward one.
    lines.append(_line("ESS:", "unavailable: not measured"))
    lines.append(_line("Mixing:", "unavailable: not measured"))
    lines.append(_line("Valid-state fraction:", _verif(verification, "task_validity")))
    lines.append(_line("Diversity:", "unavailable: not measured"))
    lines.append("")

    # -- VERDICT ------------------------------------------------------------
    energy_tv = _verif(verification, "energy_tv")
    semantic = ("?" if isinstance(energy_tv, str) else
               "✓" if abs(energy_tv) < 1e-6 else "✗")

    placement = program.get("placement")
    if placement is None:
        topological = "?"
    else:
        topological = "✓" if not placement.get("unrealized") else "✗"

    # A gate that failed its threshold but was DOWNGRADED (--allow-assumed)
    # did not block the compile -- `passed=False, downgraded=True` together
    # mean "the compile proceeded past this one under the override", not "this
    # is why the compile failed". Treating a downgraded gate the same as an
    # ordinary failure would make a genuinely COMPILED receipt's own HARDWARE
    # line read ✗, contradicting the verdict it is supposed to explain.
    hardware = ("?" if not gates else
               "✓" if all(g["passed"] or g["downgraded"] for g in gates) else "✗")

    execution_tv = _verif(verification, "execution_tv")
    floor = _verif(verification, "execution_noise_floor")
    if isinstance(execution_tv, str) or isinstance(floor, str):
        sampling = "?"
    else:
        sampling = ("✓" if execution_tv <= max(_SAMPLING_TV_MULTIPLE * floor,
                                                    _SAMPLING_TV_FLOOR) else "✗")

    checks = (("SEMANTIC", semantic), ("TOPOLOGICAL", topological),
             ("HARDWARE", hardware), ("SAMPLING", sampling))

    lines.append("VERDICT")
    for label, mark in checks:
        lines.append(f"  {mark} {label}")
    lines.append("")

    all_pass = all(mark == "✓" for _, mark in checks)
    if verdict == "COMPILED" and all_pass:
        lines.append(f"  REPRESENTATION FITS {target_name.upper()}")
    else:
        reason, remediations = _failure_narrative(passes, verification, checks, verdict)
        lines.append("Reason:")
        lines.append(reason)
        lines.append("")
        lines.append("Suggested next action:")
        if remediations:
            for r in remediations:
                lines.append(f"{r['action']}: {r['detail']}")
        else:
            lines.append("unavailable: no remediation recorded for this failure")

    return "\n".join(lines) + "\n"


def _failure_narrative(passes: dict, verification: dict, checks, verdict: str):
    """The Reason/Suggested-next-action content for a report that did not fully
    pass -- taken from the ALREADY-RECORDED GateFailure/PlacementFailure a
    rejected candidate carries (passes.json's `candidates[*].failure`, written
    by receipt.py's `_failure_dict`), never freshly composed prose.

    A rejected candidate's failure is only the reason for THIS narrative when
    the compile itself did not succeed (verdict != COMPILED) -- a losing
    candidate in a multi-candidate search (A5) is not why a COMPILED receipt
    is missing a checkmark; a verification layer whose evidence is genuinely
    unavailable (e.g. a model too large to enumerate exactly) is."""
    rejected = _rejected_candidate(passes) if verdict != "COMPILED" else None
    if rejected and rejected.get("failure"):
        f = rejected["failure"]
        cause = f.get("cause") or (
            f"{f.get('failure_class', 'failure')}: measured {f['measured']}, "
            f"limit {f['limit']}")
        return cause, f.get("remediations", ())
    if rejected:
        return rejected["reason"] or "unavailable: no reason recorded", ()

    # verdict == COMPILED but a verification-layer check still failed/unavailable
    for label, mark in checks:
        if mark != "✓":
            note_key = {"SEMANTIC": "energy_tv", "SAMPLING": "execution_tv"}.get(label)
            if note_key:
                v = verification.get(note_key)
                if isinstance(v, str):
                    return f"{label}: {v}", ()
            return f"{label} check did not pass", ()
    return "unavailable: no failing layer identified", ()
