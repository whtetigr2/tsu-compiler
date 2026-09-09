"""Task 5: the science/programmer vocabulary toggle. Pure logic, so it lives in
studio.py and is tested here rather than in Tk."""
import sys

import pytest

sys.path.insert(0, "demo")

from world.studio import label, CONTROLS


def test_every_control_has_both_vocabularies():
    """A control missing a label would render blank in one mode and nobody
    would notice until they toggled."""
    for name in CONTROLS:
        assert label(name, "science")
        assert label(name, "programmer")


def test_the_two_vocabularies_differ_for_every_control():
    for name in CONTROLS:
        assert label(name, "science") != label(name, "programmer")


def test_the_science_label_for_coupling_names_the_physics():
    assert "beta" in label("beta_j", "science").lower()


def test_an_unknown_vocabulary_is_refused():
    with pytest.raises(ValueError):
        label("beta_j", "pirate")


def test_an_unknown_control_is_refused():
    with pytest.raises(KeyError):
        label("gravity", "science")


def test_controls_covers_exactly_the_keys_the_app_asks_for():
    """Both other tests iterate CONTROLS, so neither can notice a key the app
    requires but CONTROLS lacks -- and label() raises KeyError from the layout
    builder, making that a startup crash with a green suite."""
    assert set(CONTROLS) == {
        "beta_j", "seed", "size", "sea_level", "mountain_line", "relief",
        "cost_table", "warmup"}
