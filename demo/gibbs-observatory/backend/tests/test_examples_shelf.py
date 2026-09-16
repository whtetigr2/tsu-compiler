"""The examples shelf is a curated demo shelf, not a directory listing.

Two things were wrong with it.

First, every receipt found on disk was appended to the shelf with its directory
name as its title, so a newcomer opening the application saw entries reading
`sweep_highk_biome_infeasible` and `notepad_smoke` beside the curated demos.
Those are working artifacts. They are still reachable by opening a receipt
directly; they do not belong on a shelf that is supposed to show what the
toolkit can do.

Second, `extropic: true` was a flag somebody typed into a catalog. A label is
not a fact. A workload is presented as verified here only when an oracle test
covers it and that test asserts it compiles -- so the claim on screen and the
claim the suite enforces cannot drift apart.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from backend.app.receipt_loader import examples_shelf, list_receipts  # noqa: E402
from backend.app.verified_workloads import (  # noqa: E402
    VERIFIED_WORKLOADS,
    verified_ids,
)

# Directories that exist under receipts/ and are working artifacts, not demos.
WORKING_ARTIFACTS = (
    "notepad_smoke",
    "sweep_codon_opt_tiny",
    "sweep_highk_biome_infeasible",
    "sweep_dense_qubo_clique20",
    "sweep_maxcut_cycle7",
)


def test_no_shelf_entry_is_titled_with_its_directory_name():
    for item in examples_shelf():
        assert item["title"] != item["id"], (
            f"{item['id']} is on the shelf with its directory name as its "
            f"title. Every shelf entry needs a written title, or it is a "
            f"directory listing wearing a shelf's clothes.")


def test_working_artifacts_are_not_on_the_shelf():
    ids = {item["id"] for item in examples_shelf()}
    on_disk = {r["id"] for r in list_receipts()}
    for artifact in WORKING_ARTIFACTS:
        if artifact not in on_disk:
            continue  # not present in this checkout; nothing to exclude
        assert artifact not in ids, (
            f"{artifact} is a working artifact and is being shown as an "
            f"example")


def test_sweeps_never_appear_as_examples():
    for item in examples_shelf():
        assert not item["id"].startswith("sweep_"), (
            f"{item['id']} is a parameter sweep, not a demo")


def test_verified_workloads_are_marked_verified():
    by_id = {item["id"]: item for item in examples_shelf()}
    for record in VERIFIED_WORKLOADS:
        assert record.shelf_id in by_id, (
            f"{record.shelf_id} has an oracle test proving it compiles but is "
            f"not on the shelf at all")
        item = by_id[record.shelf_id]
        assert item["verified"] is True, (
            f"{record.shelf_id} is covered by an oracle test but is not shown "
            f"as verified")
        assert item["verified_by"], (
            "a verified entry must name what verifies it")
        assert record.source in item["verified_by"] or "oracle" in (
            item["verified_by"].lower())


def test_nothing_else_claims_to_be_verified():
    """The badge is derived, so it cannot be typed onto an extra entry."""
    claimed = {item["id"] for item in examples_shelf() if item.get("verified")}
    assert claimed == set(verified_ids()), (
        f"the shelf claims these are verified: {sorted(claimed)}, but the "
        f"oracle tests cover: {sorted(verified_ids())}. The badge must be "
        f"derived from test coverage, never typed into the catalog.")


def test_extropic_workloads_are_the_verified_ones():
    """What Paul is showing off: somebody else's published work, compiling."""
    for item in examples_shelf():
        if item.get("extropic"):
            assert item["verified"] is True, (
                f"{item['id']} is presented as Extropic's published work but "
                f"no oracle test proves it still compiles. Either add the "
                f"test or stop making the claim.")


def test_every_shelf_entry_says_what_it_is():
    for item in examples_shelf():
        assert item.get("notes"), f"{item['id']} has no description"


@pytest.mark.parametrize("record", VERIFIED_WORKLOADS,
                         ids=[r.shelf_id for r in VERIFIED_WORKLOADS])
def test_each_verified_workload_has_its_spec_on_disk(record):
    spec = ROOT / "programs" / f"{record.spec_stem}.yaml"
    assert spec.is_file(), (
        f"{record.shelf_id} is advertised as a verified workload but its "
        f"program {spec} does not exist")
