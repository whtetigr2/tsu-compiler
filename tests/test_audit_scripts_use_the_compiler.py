"""Every audit script that builds an IsingModel must have DECIDED about preflight.

This project keeps building things and then not using them. `preflight()`,
`analyse_regime()`, `gate_checks()`, `ess`, `preflight.diagnostics.r_hat`,
`preflight.sweep.onsager_betac` all exist, are reviewed, and are better than the
numpy that keeps getting written next to them. At the time this test was added,
exactly ONE of sixteen audit scripts that construct an IsingModel called any of
them.

That is not because fifteen scripts were wrong to skip it -- several genuinely
have no hardware question to ask. It is because the question was never PUT. A
default of "didn't think about it" is indistinguishable in the tree from a
default of "considered and not applicable", and only one of those is a decision.

So this test does not demand that every script call preflight. It demands that
every script have an answer on the record: either it calls the tools, or it
carries a line saying why it does not. Both are fine. Silence is not.

Deliberately NOT enforced here: which tool. A script asking a graph question
wants `analyse`, one asking a hardware question wants `preflight`, one asking a
precision question wants `analyse_regime`. Encoding that choice in a test would
be guessing at intent; recording that a choice was made is not.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

AUDIT = Path(__file__).resolve().parent.parent / "audit"

# Any one of these counts as "used the compiler's own analysis".
TOOLS = (
    "preflight(",
    "analyse_regime(",
    "gate_checks(",
    "check_gates(",
    "onsager_betac(",
    "effective_sample_size(",
    "integrated_autocorrelation_time(",
    "r_hat(",
)

# The explicit opt-out. The reason is required and must say something.
OPT_OUT = re.compile(
    r"#\s*NO-PREFLIGHT:\s*(?P<reason>\S.{15,})", re.IGNORECASE)


def _audit_scripts_building_models():
    out = []
    for path in sorted(AUDIT.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "IsingModel(" in text:
            out.append((path.name, text))
    return out


SCRIPTS = _audit_scripts_building_models()


def test_there_are_audit_scripts_to_check():
    """Guard the guard: a glob that matches nothing would pass vacuously."""
    assert SCRIPTS, (
        "no audit script builds an IsingModel -- either the tree moved or this "
        "test is silently checking nothing, which is the failure mode it exists "
        "to prevent elsewhere")


@pytest.mark.parametrize("name,text", SCRIPTS, ids=[n for n, _ in SCRIPTS])
def test_script_decided_about_preflight(name, text):
    used = [t for t in TOOLS if t in text]
    opted = OPT_OUT.search(text)

    assert used or opted, (
        f"{name} builds an IsingModel without calling any of the compiler's own "
        f"analysis ({', '.join(t.rstrip('(') for t in TOOLS)}) and without "
        f"recording why.\n\n"
        f"If it has a hardware, precision or convergence question, call the "
        f"tool -- preflight(ising, target) is the front door and returns gates "
        f"with provenance, mediators, placement and node budget in one call.\n\n"
        f"If it genuinely has none, say so on one line:\n"
        f"    # NO-PREFLIGHT: <why this model is never going near hardware>\n\n"
        f"Both answers are acceptable. Not having answered is not.")


@pytest.mark.parametrize("name,text", SCRIPTS, ids=[n for n, _ in SCRIPTS])
def test_opt_out_reasons_are_reasons(name, text):
    """An opt-out that says nothing is worse than no opt-out: it looks decided."""
    m = OPT_OUT.search(text)
    if m is None:
        pytest.skip("uses the tools rather than opting out")
    reason = m.group("reason").strip()
    lazy = ("n/a", "na", "none", "not applicable", "no", "skip", "todo", "tbd")
    assert reason.lower().rstrip(".") not in lazy, (
        f"{name}'s NO-PREFLIGHT reason is {reason!r}, which records that "
        f"someone typed something, not that anyone decided anything")
