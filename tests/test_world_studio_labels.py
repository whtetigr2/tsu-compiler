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
