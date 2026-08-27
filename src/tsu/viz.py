"""Static four-layer render, from receipt data ALONE.

Layer D is rendered from `metrics.json` and `target.json`, never recomputed and
never hardcoded -- an earlier prototype printed `Oracle verdict: GREEN` as a
string literal.
"""
from __future__ import annotations

import html
import json
from pathlib import Path


def _table(d: dict) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in d.items())
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
