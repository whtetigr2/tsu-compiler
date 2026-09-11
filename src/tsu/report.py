"""The compilation report: a fixed-format text rendering of a receipt, and
NOTHING else. `render_report(receipt_dir)` reads only the JSON/text files a
receipt already contains -- it never recomputes a number the compiler did not
already record, and it never hardcodes a verdict as a string literal (that was
the exact defect viz.py's own docstring documents from an earlier prototype:
"Oracle verdict: GREEN" printed regardless of what the receipt actually said).

Any field the receipt does not contain prints `unavailable: <reason>` here --
never a blank, a zero, or an invented value. ESS and Mixing (tsu.ess) print a
real measurement when the compile's own sampling run supported a reliable
estimate, and `unavailable: <reason>` -- a short chain, or too few effective
samples relative to the estimated autocorrelation time -- when it did not;
either way this file only ever renders what verification.json/regime.json
already recorded, never a number it computed itself. Diversity (C4: distinct
valid configurations over total valid samples, plus the reachable valid
set's own size where it is enumerable, so the ratio cannot be misread on a
small state space) follows the same rule -- a real measurement when
verification has run, `unavailable: <reason>` for the reachable-set half
when the logical state space was too large to enumerate exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

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


def _coefficient_scale_display(passes: dict) -> str:
    """Task 2, note 5: a scaled sample must never be mistakable for an
    unscaled one. `coefficient_scale`/`scaled_beta` are ALWAYS present in
    passes.json (Task 2's `search.py` sets them deterministically on every
    verdict) so this only reads `unavailable: <reason>` for a receipt
    written before Task 2 existed. At `coefficient_scale == 1.0` (the
    default, and every pre-Task-2 compile) the base beta and the scaled
    beta are numerically identical, so printing "1.0 (unscaled)" is what
    keeps that case visibly distinct from a genuinely-scaled run that
    happens to land on beta==base_beta by coincidence -- there is no such
    coincidence to have at s=1.0."""
    if "coefficient_scale" not in passes:
        return "unavailable: not recorded"
    s = passes["coefficient_scale"]
    if s is None:
        return "unavailable: not recorded"
    if s == 1.0:
        return "1.0 (unscaled)"
    scaled_beta = passes.get("scaled_beta")
    if scaled_beta is None:
        return f"{_fmt(s)} (beta compensation unavailable: not recorded)"
    base_beta = scaled_beta * s
    return f"{_fmt(s)} (beta compensated: {_fmt(base_beta)} -> {_fmt(scaled_beta)})"


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
                if c["state"] in ("SEMANTICALLY_INVALID", "HARDWARE_INFEASIBLE",
                                  "COMPILER_ERROR")),
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
    # Task 6 (spec 5.3.7): what the mediation PASS itself actually did --
    # distinct from "Mediators required" above (analyse()'s theoretical
    # max-cut floor for whatever graph ended up selected, 0 once mediation
    # has already made it bipartite). `passes.json`'s "mediation" is None
    # exactly when the selected candidate never needed mediation at all.
    mediation = passes.get("mediation")
    if mediation:
        lines.append(_line("Mediation:", f"{mediation['mediator_count']:,} spin(s) "
                           f"via {mediation['partition_method']}"))
        lines.append(_line("Mediation beta:", mediation["beta_used"]))
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
    # ESS/Mixing (tsu.ess) render the real measurement when the compile's own
    # sampling run supported a reliable estimate (verification.json's `ess`,
    # regime.json's `mixing_indicator`), and `unavailable: <reason>` -- never
    # a fabricated number -- when it did not (short chain, or N/tau below
    # tsu.ess's reliability threshold).
    lines.append(_line("ESS:", _verif(verification, "ess")))
    lines.append(_line("Mixing:", _regime(regime, "mixing_indicator")))
    lines.append(_line("Valid-state fraction:", _verif(verification, "task_validity")))
    # C4: diversity -- distinct valid configurations over total valid samples
    # need no exact reference and are always real once verification has run
    # (same as task_validity); the reachable-valid-set SIZE is the number
    # that makes that ratio readable rather than misleading on a small state
    # space, and is unavailable-with-a-reason only when the logical state
    # space was too large to enumerate exactly.
    distinct = _verif(verification, "diversity_distinct")
    if isinstance(distinct, str):
        diversity_line = distinct
    else:
        valid_n = verification.get("diversity_valid_samples")
        reachable = _verif(verification, "diversity_reachable")
        tail = f"{reachable}" if isinstance(reachable, str) else \
            f"{reachable} reachable valid state(s)"
        diversity_line = f"{distinct}/{valid_n if valid_n is not None else '?'} " \
                         f"valid samples distinct ({tail})"
    lines.append(_line("Diversity:", diversity_line))
    lines.append(_line("Coefficient scale:", _coefficient_scale_display(passes)))
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


# ============================================================================
# C3: `tsu explain` -- the fuller, teaching-trace rendering. `tsu report`
# above stays the terse form; this is additive, not a replacement. Same
# honesty rules apply: rendered from receipt data alone, never recomputed,
# never hardcoded; any field the receipt does not contain prints
# `unavailable: <reason>`. Where this can state WHY a choice was made, it is
# sourced from recorded evidence already in the receipt (candidates.json's
# rows, passes.json's `ordering_rationale`, gates.json's remediations,
# formulation.json's per-construct records) -- never freshly invented prose
# about a specific compile's numbers.
#
# G1: the FORMULATION layer used to fall back on a small STATIC glossary of
# what each term KIND *is* mathematically -- informative about the shape, but
# silent on WHY that shape was chosen for THIS spec. It now renders
# formulation.json instead: structured `FormulationRationale` records built
# by spec.py itself at term-expansion time (never reconstructed here), each
# carrying the construct, the IR shape it became, how many terms and over
# what, and -- when the construct has one -- the structural reason. A
# construct with no reason to report (a hand-written `linear`/`product` term)
# renders that absence explicitly, never invented prose standing in for one.
# ============================================================================


def _yaml_safe_load(text: str) -> dict:
    try:
        return yaml.safe_load(text) or {}
    except Exception:
        return {}


def _contract_rule_counts(spec_raw: dict) -> dict:
    counts: dict[str, int] = {}
    for r in (spec_raw.get("contract") or {}).get("validate", []) or []:
        k = r.get("rule", "?")
        counts[k] = counts.get(k, 0) + 1
    return counts


def _candidate_field(v):
    return "unavailable" if v is None else _fmt(v)


def _magnitude_range(values, verdict: str, label: str) -> str:
    if not values:
        if verdict != "COMPILED":
            return _as_unavailable(f"representation was never reached (verdict={verdict})")
        return f"unavailable: no {label} in this model"
    mags = [abs(v) for v in values]
    return f"[{_fmt(min(mags))}, {_fmt(max(mags))}]"


def _sample_headline(sample: dict, verdict: str) -> str:
    """G2: the one-line summary for `sample.json`'s decoded result -- sourced
    from the SAME record `search.py`'s `_verify` persisted, never
    recomputed. Three honest outcomes: a real solution (codeword + task-
    valid), a FAILING sample (codeword but task-invalid, itemised below),
    or -- when not even one codeword was drawn -- the recorded note, never a
    blank or a fabricated stand-in."""
    if not sample or sample.get("decoded") is None:
        if sample and sample.get("note"):
            return _as_unavailable(sample["note"])
        if verdict != "COMPILED":
            return _as_unavailable(f"no sample recorded (verdict={verdict})")
        return "unavailable: not recorded"
    return ("valid codeword, task-valid" if sample["task_valid"] else
           "valid codeword, TASK-INVALID (see violations below)")


def render_explain(receipt_dir) -> str:
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
    energy = _load(d, "energy.json")
    cost = _load(d, "cost.json")
    formulation = _load(d, "formulation.json")
    if not isinstance(formulation, list):
        formulation = []
    sample = _load(d, "sample.json")
    spec_yaml_path = d / "spec.yaml"
    spec_raw = _yaml_safe_load(spec_yaml_path.read_text(encoding="utf-8")) \
        if spec_yaml_path.exists() else {}

    verdict = passes.get("verdict", "unavailable")

    def _eline(label: str, value: Any) -> str:
        return _line(label, value, width=38)

    lines = ["TSU COMPILATION EXPLAIN", "=" * 40, ""]

    # -- APPLICATION (1): what the spec declared, in its own vocabulary -----
    lines.append("APPLICATION")
    lines.append(_eline("Spec name:", spec_raw.get("name") or
                       "unavailable: not recorded"))
    description = (spec_raw.get("description") or "").strip()
    lines.append(_eline("Description:",
                       description.splitlines()[0] if description else
                       "unavailable: not recorded in spec.yaml"))
    lines.append(_eline("Variables:",
                       workload.get("variables", "unavailable: not recorded")))
    lines.append(_eline("State cardinality:",
                       workload.get("state_cardinality", "unavailable: not recorded")))
    lines.append(_eline("Constraint rule kinds:",
                       workload.get("constraint_classes", "unavailable: not recorded")))
    lines.append("")

    # -- FORMULATION: the constraints restated as energy terms, and WHY -----
    lines.append("FORMULATION")
    if "generate" in spec_raw:
        lines.append(_eline("Generated shape:",
                           spec_raw["generate"].get("kind", "unavailable")))
    lines.append(_eline("Declared terms:",
                       workload.get("logical_interactions", "unavailable: not recorded")))
    # G1: each record is spec.py's OWN account of why a construct's terms took
    # the IR shape they did, sourced from formulation.json (built at term-
    # expansion time), never reconstructed here. `scope` is only printed when
    # the record actually has one (a single hand-written term has none).
    if formulation:
        for r in formulation:
            scope = f" over {r['scope']}" if r.get("scope") else ""
            lines.append(f"    {r['construct']} x{r['count']}{scope} "
                        f"-- {r['ir_shape']}")
            reason = r.get("reason") or ""
            lines.append(f"        why: {reason}" if reason else
                        "        why: (none recorded -- this shape is a "
                        "direct hand-written declaration, not a template's "
                        "design decision)")
    elif workload.get("logical_interactions") == 0:
        lines.append("    no terms declared in this spec")
    else:
        lines.append("    unavailable: no term-formulation rationale recorded "
                    "in this receipt")
    rule_counts = _contract_rule_counts(spec_raw)
    for kind in sorted(rule_counts):
        lines.append(f"    contract rule {kind} x{rule_counts[kind]}")
    lines.append("")

    # -- REPRESENTATION: encoding, candidates considered, why this one won --
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

    lines.append("REPRESENTATION")
    lines.append(_eline("Encoding selected:", encoding_display))
    lines.append(_eline("Logical variables:", n_nodes_display))
    lines.append(_eline("Logical edges:", n_edges_display))
    lines.append(_eline("Graph:", graph_display))
    if candidates_table:
        lines.append(_eline("Candidates considered:", len(candidates_table)))
        for row in candidates_table:
            lines.append(
                f"    {row['encoding']:<12} {row['state']:<22} "
                f"pbits={_candidate_field(row.get('physical_pbits'))} "
                f"colour_blocks={_candidate_field(row.get('colour_blocks'))} "
                f"|J|max={_candidate_field(row.get('max_abs_J'))}")
    else:
        lines.append("  Candidates considered: unavailable: not recorded")
    lines.append(_eline("Selection rationale:",
                       passes.get("ordering_rationale") or "unavailable: not recorded"))
    lines.append("")

    # -- ENERGY: term inventory, |J|/|b| ranges ------------------------------
    term_inv = energy.get("term_counts")
    if term_inv:
        linear_n, product_n = term_inv.get("linear", 0), term_inv.get("product", 0)
        inventory = f"linear={linear_n}, product={product_n} " \
                   f"(total {energy.get('n_terms', linear_n + product_n)})"
    elif verdict != "COMPILED":
        inventory = _as_unavailable(f"representation was never reached (verdict={verdict})")
    else:
        inventory = "unavailable: not recorded"
    lines.append("ENERGY")
    lines.append(_eline("Term inventory:", inventory))
    lines.append(_eline("|J| range:",
                       _magnitude_range(program.get("weights") or [], verdict, "couplings")))
    lines.append(_eline("|b| range:",
                       _magnitude_range(program.get("biases") or [], verdict, "biases")))
    lines.append("")

    # -- TOPOLOGY: logical graph, degree, bipartiteness, connectivity -------
    mediators = _rep_display(metrics, rep_row, verdict, "mediators", "mediators",
                             "mediators_note")
    lines.append("TOPOLOGY")
    lines.append(_eline("Logical spins:", n_nodes_display))
    lines.append(_eline("Logical edges:", n_edges_display))
    lines.append(_eline("Max degree:", _metric(metrics, verdict, "max_degree")))
    lines.append(_eline("Graph:", graph_display))
    lines.append(_eline("Connectivity residual (|E| - max-cut):", mediators))
    lines.append("")

    # -- PHYSICAL MAPPING: placement, mediators, p-bits, colour blocks ------
    n_edges_raw, n_edges_fb = _rep_raw(metrics, rep_row, "n_edges", "logical_edges")
    n_nodes_raw, n_nodes_fb = _rep_raw(metrics, rep_row, "n_nodes", "logical_spins")
    mediators_raw, mediators_fb = _rep_raw(metrics, rep_row, "mediators", "mediators")
    if isinstance(mediators_raw, int) and n_nodes_raw is not None:
        physical_pbits = _fmt(n_nodes_raw + mediators_raw)
    else:
        physical_pbits = mediators
    placement = program.get("placement")
    lines.append("PHYSICAL MAPPING")
    lines.append(_eline("Mediators inserted:", mediators))
    lines.append(_eline("Physical p-bits:", physical_pbits))
    lines.append(_eline("Colour blocks:",
                       _rep_display(metrics, rep_row, verdict, "colour_blocks",
                                    "colour_blocks")))
    # Task 6 (spec 5.3.7): the mediation PASS's own record -- how many
    # mediator spins it actually inserted, by what method, and at what beta
    # (spec 5.3.5: those couplings are only valid at this beta). None when
    # the selected candidate never needed mediation.
    mediation = passes.get("mediation")
    lines.append(_eline("Mediation method:",
                       mediation["partition_method"] if mediation else
                       "unavailable: not mediated"))
    lines.append(_eline("Mediation beta:",
                       mediation["beta_used"] if mediation else
                       "unavailable: not mediated"))
    if placement is None:
        placement_line = _as_unavailable(f"no placement recorded (verdict={verdict})")
        lines.append(_eline("Placement:", placement_line))
    else:
        lines.append(_eline("Placed nodes:", len(placement.get("coords", {}))))
        lines.append(_eline("Realized edges:", len(placement.get("realized", []))))
        lines.append(_eline("Unrealized edges:", len(placement.get("unrealized", []))))
    lines.append("")

    # -- SAMPLER: kernel, schedule, blocks, and the clamp (C1) if any -------
    schedule = program.get("schedule")
    kernel = _KERNEL_LABELS.get(schedule, schedule) if schedule else \
        _as_unavailable(f"no program built (verdict={verdict})")
    blocks = program.get("blocks")
    lines.append("SAMPLER")
    lines.append(_eline("Kernel:", kernel))
    lines.append(_eline("Schedule:", schedule or
                       _as_unavailable(f"no program built (verdict={verdict})")))
    if blocks:
        sizes = [len(b) for b in blocks]
        lines.append(_eline("Free blocks:", f"{len(blocks)} (sizes {sizes})"))
    else:
        lines.append(_eline("Free blocks:",
                           _as_unavailable(f"no program built (verdict={verdict})")
                           if verdict != "COMPILED" else "0"))
    lines.append(_eline("Coefficient scale:", _coefficient_scale_display(passes)))
    clamp = program.get("clamp") or {}
    lines.append(_eline("Clamp:", clamp if clamp else "none"))
    sampler_cost = cost.get("sampler") or {}
    wall_time = sampler_cost.get("wall_time_s")
    sps = sampler_cost.get("samples_per_second")
    lines.append(_eline("Sampler wall time (s):",
                       wall_time if wall_time is not None else "unavailable: not recorded"))
    lines.append(_eline("Samples/second:",
                       sps if sps is not None else "unavailable: not recorded"))
    lines.append("")

    # -- VERIFICATION: each layer's equivalence result, or unavailable ------
    lines.append("VERIFICATION")
    lines.append(_eline("Energy equivalence (TV):", _verif(verification, "energy_tv")))
    lines.append(_eline("Execution TV:", _verif(verification, "execution_tv")))
    lines.append(_eline("Execution noise floor:",
                       _verif(verification, "execution_noise_floor")))
    lines.append(_eline("Cross-check TV (torx):", _verif(verification, "cross_check_tv")))
    lines.append(_eline("Codeword violation rate:",
                       _verif(verification, "codeword_violation_rate")))
    lines.append(_eline("ESS:", _verif(verification, "ess")))
    lines.append(_eline("Mixing (IAT):", _regime(regime, "mixing_indicator")))
    lines.append(_eline("Diversity (distinct valid):",
                       _verif(verification, "diversity_distinct")))
    lines.append(_eline("Diversity (reachable valid states):",
                       _verif(verification, "diversity_reachable")))
    baseline = cost.get("baseline") or {}
    exact_wall_time = baseline.get("exact_wall_time_s")
    if exact_wall_time is not None:
        lines.append(_eline("CPU baseline exact wall time (s):", exact_wall_time))
        lines.append(_eline("CPU baseline sampler wall time (s):",
                           baseline.get("sampler_wall_time_s", "unavailable: not recorded")))
    else:
        lines.append(_eline("CPU baseline exact wall time (s):",
                           _as_unavailable(baseline.get("note") or "not recorded")))
    if baseline.get("disclaimer"):
        lines.append(f"    ({baseline['disclaimer']})")
    lines.append("")

    # -- APPLICATION (2): the decoded result and task-level validity --------
    lines.append("APPLICATION")
    lines.append(_eline("Task validity:", _verif(verification, "task_validity")))
    lines.append(_eline("Decoded result:", _sample_headline(sample, verdict)))
    if sample.get("decoded") is not None:
        for name, value in sample["decoded"].items():
            lines.append(f"    {name} = {_fmt(value)}")
        lines.append(_eline("Codeword:", sample["is_codeword"]))
        lines.append(_eline("Task-valid:", sample["task_valid"]))
        for v in sample.get("violations") or ():
            lines.append(f"    violation: {v}")
        lines.append(_eline("Sample seed:", sample["seed"]))
        lines.append(_eline("Clamp at draw time:", sample["clamp"] or "none"))
    elif sample.get("note"):
        lines.append(f"    {sample['note']}")

    return "\n".join(lines) + "\n"
