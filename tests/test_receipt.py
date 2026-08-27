import json
from pathlib import Path

from tsu.spec import load_spec
from tsu.target import Z1
from tsu.passes.search import compile_spec
from tsu.receipt import write_receipt, load_receipt, replay


def test_receipt_contains_every_required_artifact(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    for name in ("spec.yaml", "spec.sha256", "target.json", "passes.json",
                 "gates.json", "metrics.json", "verification.json",
                 "program.json", "environment.json", "regime.json"):
        assert (Path(d) / name).exists(), f"missing {name}"


def test_target_json_preserves_provenance_including_assumed(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    t = json.loads((Path(d) / "target.json").read_text())
    assert t["max_abs_coupling"]["source"] == "assumed"
    assert t["degree"]["source"] == "F-14"


def test_receipt_replays_and_hashes_match(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    r = replay(d)
    assert r.matches is True, f"replay diverged: {r.diffs}"


def test_verification_never_writes_a_number_it_does_not_have(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    v = json.loads((Path(d) / "verification.json").read_text())
    for value in v.values():
        assert not (isinstance(value, str) and value == ""), \
            "an empty string is not a verdict; write the number or 'unavailable: reason'"
