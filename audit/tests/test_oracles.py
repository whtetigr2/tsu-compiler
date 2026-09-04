"""audit/tests/test_oracles.py -- TDD test suite for audit/oracles (Task A4).

Run with the project's pinned Python 3.14 interpreter, from the repo root:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
        -m pytest audit/tests/test_oracles.py -v

These tests exercise the INDEPENDENT oracles under audit/oracles/. Per plan
Task A4 ("The independence is the entire point"), audit/oracles/exact.py
writes the physics from scratch and does NOT import src/tsu -- an oracle
sharing code with the thing under test verifies nothing. This test file
itself also never imports src/tsu.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

# audit/ (the parent of oracles/), not the repo root -- `import oracles.exact`
# below resolves against THIS directory, matching the plan's stated expected
# first-run failure "ModuleNotFoundError: No module named 'oracles'".
AUDIT_DIR = Path(__file__).resolve().parents[1]
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))

from oracles.exact import exact_boltzmann, exact_energy  # noqa: E402
from oracles.wolfram import wolfram_query  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1 test, verbatim from the plan (2026-09-03-lattice-pre-theme-audit.md,
# Task A4, Step 1). This is the one whose FIRST run's exact failure message
# was recorded before any implementation existed -- see audit/provenance.md
# / the task's final report for that recorded message.
# ---------------------------------------------------------------------------

def test_exact_boltzmann_matches_hand_computation():
    """Two spins, J=+1 ferromagnetic, no field, beta=1. Hand-computed:
    aligned states have E=-1, anti-aligned E=+1 (E = -J*s_i*s_j), so
    p_aligned/p_anti = exp(2) exactly."""
    states, probs = exact_boltzmann(J={(0, 1): 1.0}, b=[0.0, 0.0], beta=1.0)
    p = {tuple(s): pr for s, pr in zip(states, probs)}
    aligned = p[(1, 1)] + p[(0, 0)]
    anti = p[(1, 0)] + p[(0, 1)]
    assert math.isclose(aligned / anti, math.exp(2.0), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Supplementary coverage for exact.py -- not literally in the plan's Step 1
# snippet, but exercising exact_energy directly (the plan's separate
# `exact_energy(state, J, b) -> float` interface) and basic well-formedness
# of the returned distribution.
# ---------------------------------------------------------------------------

def test_exact_boltzmann_probabilities_sum_to_one():
    states, probs = exact_boltzmann(J={(0, 1): 1.0}, b=[0.3, -0.2], beta=0.7)
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-12)
    assert len(states) == 4  # 2 spins -> 2**2 configurations, none pruned


def test_exact_energy_matches_hand_computation_for_aligned_and_antialigned():
    # E = -J*s0*s1 - b0*s0 - b1*s1, no field here.
    assert math.isclose(exact_energy((1, 1), {(0, 1): 1.0}, [0.0, 0.0]), -1.0)
    assert math.isclose(exact_energy((1, 0), {(0, 1): 1.0}, [0.0, 0.0]), 1.0)


def test_exact_energy_accepts_spin_valued_state_as_well_as_occupancy():
    """The plan's convention is s in {-1,+1}; callers commonly hold a
    {0,1} occupancy vector instead (the compiler's own convention -- see
    src/tsu/passes/lower.py's docstring, read but not imported here).
    exact_energy must treat (1, 0) [occupancy] and (1, -1) [spin] as the
    SAME physical state."""
    e_from_occupancy = exact_energy((1, 0), {(0, 1): 1.0}, [0.0, 0.0])
    e_from_spin = exact_energy((1, -1), {(0, 1): 1.0}, [0.0, 0.0])
    assert e_from_occupancy == e_from_spin == 1.0


def test_exact_energy_rejects_a_state_entry_that_is_neither_01_nor_pm1():
    with pytest.raises(ValueError):
        exact_energy((1, 2), {(0, 1): 1.0}, [0.0, 0.0])


def test_exact_boltzmann_requires_at_least_one_spin():
    with pytest.raises(ValueError):
        exact_boltzmann(J={}, b=[], beta=1.0)


def test_exact_boltzmann_sign_flip_of_j_diverges_the_energies():
    """A falsification test in the spirit of R3's mandatory sign-flip check
    (plan Wave 2): flipping J's sign must swap which pair of states is
    favoured, not leave the ratio unchanged. A test that passes under both
    signs would be testing nothing."""
    states, probs_pos = exact_boltzmann(J={(0, 1): 1.0}, b=[0.0, 0.0], beta=1.0)
    _, probs_neg = exact_boltzmann(J={(0, 1): -1.0}, b=[0.0, 0.0], beta=1.0)
    p_pos = {tuple(s): pr for s, pr in zip(states, probs_pos)}
    p_neg = {tuple(s): pr for s, pr in zip(states, probs_neg)}
    aligned_pos = p_pos[(1, 1)] + p_pos[(0, 0)]
    aligned_neg = p_neg[(1, 1)] + p_neg[(0, 0)]
    assert aligned_pos > 0.5  # ferromagnetic: aligned states favoured
    assert aligned_neg < 0.5  # antiferromagnetic: aligned states disfavoured
    assert not math.isclose(aligned_pos, aligned_neg)


# ---------------------------------------------------------------------------
# wolfram.py -- the independence contract: never raises, never fabricates a
# local number dressed as a third-party confirmation. If the key or the
# network is unavailable, the reason is surfaced verbatim.
# ---------------------------------------------------------------------------

def test_wolfram_query_never_raises_and_returns_a_labelled_string():
    result = wolfram_query("2+2")
    assert isinstance(result, str) and result
    if result.startswith("UNAVAILABLE"):
        assert result.startswith("UNAVAILABLE: ") and len(result) > len("UNAVAILABLE: ")


def test_wolfram_appid_is_discovered_without_ever_being_printed_or_asserted_against():
    """Only checks that SOME key was located when the vault file exists in
    this environment -- the value itself is never compared, printed, or
    written by this test (see wolfram.py's own module docstring / CLAUDE.md
    Credentials rules)."""
    from oracles.wolfram import _read_wolfram_appid, SECRETS_PATH
    if not SECRETS_PATH.exists():
        pytest.skip("secrets.local.md not present in this environment")
    appid = _read_wolfram_appid()
    assert appid is not None and len(appid) > 0


def test_wolfram_cross_checks_exact_oracle_on_the_two_spin_case():
    """Task A4 Step 6: cross-check the oracle against Wolfram on the plan's
    two-spin case, side by side. exact_boltzmann's own aligned/anti-aligned
    ratio (locally computed) is compared against Wolfram's independent
    evaluation of exp(2) (third-party computed) -- two different code paths,
    same physical quantity. If Wolfram is unreachable this is recorded via
    skip with the precise UNAVAILABLE reason, never silently treated as a
    pass or papered over with a local fallback."""
    states, probs = exact_boltzmann(J={(0, 1): 1.0}, b=[0.0, 0.0], beta=1.0)
    p = {tuple(s): pr for s, pr in zip(states, probs)}
    local_ratio = (p[(1, 1)] + p[(0, 0)]) / (p[(1, 0)] + p[(0, 1)])

    # "N[exp(2), 12]" (a decimal-approximation request), not bare "exp(2)"
    # -- Wolfram's "Result" pod for the bare symbolic query returns "e^2" as
    # plaintext, not a decimal, which is not comparable to a float without
    # re-evaluating it ourselves (exactly the local-fallback this module
    # must never do).
    wolfram_result = wolfram_query("N[exp(2), 12]")
    print(f"\n[A4 two-spin cross-check] exact_boltzmann aligned/anti-aligned "
          f"ratio = {local_ratio!r}")
    print(f"[A4 two-spin cross-check] Wolfram Alpha N[exp(2), 12] "
          f"= {wolfram_result!r}")

    if wolfram_result.startswith("UNAVAILABLE"):
        pytest.skip(f"Wolfram Alpha unreachable: {wolfram_result}")

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", wolfram_result)
    if not match:
        pytest.skip(f"Wolfram Alpha response not numeric-parseable: {wolfram_result!r}")
    wolfram_value = float(match.group(0))
    assert math.isclose(local_ratio, wolfram_value, rel_tol=1e-4)
