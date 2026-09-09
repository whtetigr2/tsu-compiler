"""Task 4: saving and restoring a run."""
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.studio import Studio, Sampled, Readout
from world.run_log import save_run, load_run


@pytest.fixture(scope="module")
def studio():
    s = Studio()
    s.generate(Sampled(seed=21, size=64, warmup=200))
    s.set_readout(Readout(sea_level=0.15, mountain_line=-0.1, relief=1.3,
                          cost_table="wide"))
    return s


def test_save_writes_the_expected_files(tmp_path, studio):
    names = {p.name for p in save_run(studio, tmp_path)}
    assert {"world.png", "world.txt", "world.json",
            "studio.json", "provenance.json", "README.md"} <= names


def test_a_saved_run_restores_every_slider(tmp_path, studio):
    """A run you cannot restore exactly is a screenshot, not a log."""
    save_run(studio, tmp_path)
    sampled, readout = load_run(tmp_path)
    assert sampled == studio.sampled
    assert readout == studio.readout


def test_provenance_records_the_jax_backend(tmp_path, studio):
    """thrml SIMULATES the sampling a Z1 would do, on CPU here. A report that
    omits the backend invites the reader to assume hardware was involved.

    The value is checked, not merely its truthiness: `_versions()` falls back to
    the string "unknown" on any failure, which is truthy, so a truthiness check
    would pass on a silently-broken read -- and "unknown" does not record the
    backend, which is the thing the artifact exists to state."""
    save_run(studio, tmp_path)
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["jax_backend"] in ("cpu", "gpu", "tpu")
    assert prov["thrml"] != "unavailable"
    assert prov["jax"] != "unavailable"


def test_studio_json_records_the_resolved_tables_not_just_the_slider_names(
        tmp_path, studio):
    """The slider values alone are not enough to check the work: the reader
    needs the band bounds and costs those sliders actually resolved to."""
    save_run(studio, tmp_path)
    d = json.loads((tmp_path / "studio.json").read_text(encoding="utf-8"))
    assert len(d["resolved_bounds"]) == 5
    assert len(d["resolved_costs"]) == 6
    assert d["readout"]["cost_table"] == "wide"


def test_restored_run_reproduces_the_same_world(tmp_path, studio):
    """The strongest claim the log makes. Same seed, same sliders, same world."""
    save_run(studio, tmp_path)
    sampled, readout = load_run(tmp_path)
    fresh = Studio()
    fresh.generate(sampled)
    v = fresh.set_readout(readout)
    assert np.array_equal(v.terrain, studio.view().terrain)


def test_readme_states_which_diagnostics_were_not_run(tmp_path, studio):
    """A report that silently omits the sampler panel invites the reader to
    assume it passed. Plan A cannot produce it, so the README says so."""
    save_run(studio, tmp_path)
    txt = (tmp_path / "README.md").read_text(encoding="utf-8").lower()
    assert "not" in txt and ("diagnost" in txt or "autocorrelation" in txt)


def test_the_export_reflects_the_sliders_not_the_original_world(tmp_path, studio):
    """save_run must export what is ON SCREEN. An earlier version exported
    `studio.world` -- the world as generated, before any slider moved -- so a
    user who reshaped their world and saved got a picture of the world they
    started with. The round-trip test cannot catch this, because it compares
    two views rather than the written file."""
    studio.set_readout(Readout(sea_level=-0.45))
    save_run(studio, tmp_path)
    dry = json.loads((tmp_path / "world.json").read_text(encoding="utf-8"))["terrain"]
    studio.set_readout(Readout(sea_level=+0.45))
    save_run(studio, tmp_path)
    wet = json.loads((tmp_path / "world.json").read_text(encoding="utf-8"))["terrain"]
    assert dry != wet, "the exported world must change when a readout slider moves"
    studio.set_readout(Readout(sea_level=0.15, mountain_line=-0.1, relief=1.3,
                               cost_table="wide"))
