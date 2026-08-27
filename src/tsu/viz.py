"""Static four-layer render, from receipt data ALONE.

Layer D is rendered from `metrics.json` and `target.json`, never recomputed and
never hardcoded -- an earlier prototype printed `Oracle verdict: GREEN` as a
string literal.

I5/I8 (final review): several fields are legitimately absent (a sentinel like
`mediators == -1` "not computed", or a genuine `null` like `execution_noise_floor`
when there is no exact reference) and every one of them has an explanatory note
sitting right next to it in the same JSON -- but this renderer used to print the
bare value, so an absent field showed up as the literal string "None" with no
reason visible anywhere on the page. `_resolve_notes` folds a `<field>_note`
sibling into `<field>` whenever `<field>` itself is null, so the receipt always
shows a reason instead of a bare `None`.
"""
from __future__ import annotations

import html
import json
from pathlib import Path


def _resolve_notes(d: dict) -> dict:
    """For every `<field>_note` key, if `<field>` is None, replace it with the
    note text and drop the separate `_note` row. Leaves everything else alone."""
    notes = {k[:-len("_note")]: v for k, v in d.items() if k.endswith("_note")}
    out = {}
    for k, v in d.items():
        if k.endswith("_note"):
            continue
        out[k] = notes[k] if v is None and k in notes else v
    return out


def _table(d: dict) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in _resolve_notes(d).items())
    return f"<table>{rows}</table>"


def render(receipt_dir, out_file) -> Path:
    d = Path(receipt_dir)
    load = lambda n: json.loads((d / n).read_text())
    target, metrics = load("target.json"), load("metrics.json")
    verification, regime, passes = (load("verification.json"), load("regime.json"),
                                    load("passes.json"))
    spec_text = (d / "spec.yaml").read_text(encoding="utf-8")

    hw = {k: f"{v['value']}  [source: {v['source']}]"
          for k, v in target.items() if isinstance(v, dict)}

    body = f"""<h1>TSU compilation receipt</h1>
<p class="disclaim">Simulated on host CPU. No hardware, speed or energy claim.
Only the block-Gibbs sampling loop would ever execute on a TSU, and it did not.</p>
<h2>A. Problem space</h2><pre>{html.escape(spec_text)}</pre>
<h2>B. Representation</h2>{_table(metrics)}
<p>verdict: <b>{html.escape(str(passes.get('verdict')))}</b> &middot;
ideal control passed: {passes.get('ideal_passed')}</p>
<h2>C. Thermodynamic state</h2>{_table(verification)}{_table(regime)}
<h2>D. Hardware mapping</h2>{_table(hw)}"""

    css = ("body{font:14px/1.5 system-ui;margin:2rem;max-width:60rem}"
           "table{border-collapse:collapse;margin:.5rem 0}"
           "th,td{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}"
           "th{background:#f4f4f4;font-weight:600}"
           "pre{background:#f7f7f7;padding:.8rem;overflow-x:auto}"
           ".disclaim{color:#a15c00;font-size:13px}")
    out = Path(out_file)
    out.write_text(f"<!doctype html><meta charset='utf-8'><title>TSU receipt</title>"
                   f"<style>{css}</style>{body}", encoding="utf-8")
    return out
