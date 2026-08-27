"""Spec section 12, criteria 1-7 plus 6b-6d, as executable assertions."""
import json
from pathlib import Path

import numpy as np
import pytest

from tsu.cli import main
from tsu.spec import load_spec
from tsu.target import Z1
from tsu.passes.search import compile_spec
from tsu.passes.encode import encode
from tsu.receipt import replay, write_receipt


def test_1_compile_produces_a_receipt(tmp_path):
    assert main(["compile", "specs/toy.yaml", "--target", "z1",
                 "--out", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "program.json").exists()


def test_2_replay_reproduces_it(tmp_path):
    main(["compile", "specs/toy.yaml", "--target", "z1", "--out", str(tmp_path / "r")])
    assert replay(tmp_path / "r").matches is True


def test_3_both_stacks_are_exercised_separately(tmp_path):
    """thrml carries the reference for toy.yaml. The torx route is exact only for a
    single-bond model, so toy.yaml (4 nodes, 3 edges) correctly reports the
    cross-check as unavailable -- composing PISING across edges is a Trotter
    splitting with its own error, and a model torx cannot do exactly must not be
    handed a torx number it did not produce. torx is instead exercised directly
    on a two-node single-edge model, which IS the case the route is exact for --
    that is what proves both stacks are used, separately."""
    from tsu.backends.torx_backend import torx_cross_check
    from tsu.passes.lower import IsingModel

    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verification.energy_tv is not None, "thrml IsingEBM.energy reference missing"
    assert c.verification.cross_check_tv is None
    assert "Trotter" in c.verification.cross_check_note

    bond = IsingModel(("a", "b"), ((0, 1),), np.array([-0.5]), np.zeros(2), 1.0, 0.0)
    tx, note = torx_cross_check(bond)
    assert tx is not None, f"torx must be genuinely exercised: {note}"


def test_4_sampled_matches_exact_within_the_reported_noise_floor():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verification.execution_tv < max(
        5 * c.verification.execution_noise_floor, 0.05)


def test_5_every_hard_gate_has_a_test_that_fires_it(tmp_path):
    assert compile_spec(load_spec("specs/too_dense.yaml"), Z1).verdict == "HARDWARE"
    assert compile_spec(load_spec("specs/too_strong.yaml"), Z1).verdict == "HARDWARE"


def test_6b_ideal_control_reports_LOGICAL_not_hardware():
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    assert c.verdict == "LOGICAL"
    assert c.hardware_evaluated is False


def test_6c_decode_round_trips_all_twelve_states_and_validate_names_violations():
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    assert sum(1 for _ in s.assignments()) == 12
    for asg in s.assignments():
        assert enc.decode(enc.encode_assignment(asg)) == asg
    bad = s.contract.validate({"a": 1, "b": 1, "c": 0})
    assert len(bad.violations) == 2


def test_6d_regime_report_populated_and_honest_about_what_it_did_not_measure(tmp_path):
    """toy.yaml's chain mixes fast enough (see tsu.ess/test_ess.py and
    test_verify.py) that mixing_indicator is now a REAL number, not the old
    permanent "unmeasured" -- tsu.ess wiring means this compiler can actually
    measure it here. `energy_scale` is this vertical slice's remaining
    honestly-unmeasured cheap field (regime.py's own docstring: "cheap fields
    only"), so it is what now demonstrates the "honest about what it did not
    measure" half of this test's name."""
    main(["compile", "specs/toy.yaml", "--target", "z1", "--out", str(tmp_path / "r")])
    r = json.loads((tmp_path / "r" / "regime.json").read_text())
    assert r["coupling_utilisation"] is not None
    assert isinstance(r["mixing_indicator"], float)
    assert r["energy_scale"] is None


def test_8_a_zero_edge_model_compiles_end_to_end_instead_of_crashing(tmp_path):
    """C3: the simplest legal workload (one binary variable, one linear term, no
    products) lowers to an IsingModel with zero edges. thrml's own
    IsingEBM.factors builds a SpinEBMFactor over that empty edge block and raises
    a raw `IndexError: tuple index out of range` -- inside `_verify`, AFTER the
    verdict was already COMPILED, so nothing catches it: no receipt, no verdict,
    a vendor stack trace. Compiling this spec end to end (through the CLI, which
    is where a raw exception would otherwise surface) must produce a receipt
    with a verdict, not a crash."""
    rc = main(["compile", "specs/edgeless.yaml", "--target", "z1",
               "--out", str(tmp_path / "r")])
    assert rc == 0
    passes = json.loads((tmp_path / "r" / "passes.json").read_text())
    assert passes["verdict"] == "COMPILED"
    verification = json.loads((tmp_path / "r" / "verification.json").read_text())
    assert verification["energy_tv"] is not None, \
        "the zero-edge exact reference must actually run, not silently skip"
    assert verification["task_validity"] is not None


def test_7_visualize_renders_four_layers_from_receipt_data(tmp_path):
    main(["compile", "specs/toy.yaml", "--target", "z1", "--out", str(tmp_path / "r")])
    html = tmp_path / "r.html"
    main(["visualize", str(tmp_path / "r"), "--out", str(html)])
    text = html.read_text(encoding="utf-8")
    for layer in ("Problem space", "Representation", "Thermodynamic state",
                  "Hardware mapping"):
        assert layer in text
