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


def test_the_api_carries_the_material_the_detail_pane_renders():
    """Data only the API can see is not shipped.

    The shelf gained each verified workload's problem, math, measured hardware
    cost and citation, and for a while none of it reached the screen. These
    fields are the contract the picker's detail pane renders, so losing one
    silently empties a section of the UI rather than failing anywhere.
    """
    from fastapi.testclient import TestClient

    from backend.app.main import app

    payload = TestClient(app).get("/api/examples").json()
    items = {e["id"]: e for e in payload["examples"]}

    for record in VERIFIED_WORKLOADS:
        item = items[record.shelf_id]
        for field in ("plain_name", "one_line", "problem", "math", "hardware",
                      "citation", "verified_by", "published_by"):
            assert item.get(field), (
                f"{record.shelf_id} reaches the UI without {field}; the detail "
                f"pane would render an empty section")
        assert item["verified"] is True
        assert item["plain_name"] != item["id"], (
            "the picker leads with plain_name, which must not be the "
            "directory name")


def test_unverified_examples_still_carry_a_name_and_a_description():
    """The detail pane falls back to these, so they cannot be empty."""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    for item in TestClient(app).get("/api/examples").json()["examples"]:
        assert item.get("plain_name"), f"{item['id']} has no plain_name"
        assert item.get("notes"), f"{item['id']} has no description"


@pytest.mark.parametrize("record", VERIFIED_WORKLOADS,
                         ids=[r.shelf_id for r in VERIFIED_WORKLOADS])
def test_the_prose_matches_the_receipt_the_user_actually_opens(record):
    """Two numbers describing the same thing must not disagree on screen.

    Caught by looking at the rendered picker. Beside each name it shows the
    receipt's spin count, and the detail pane shows a written description of the
    hardware cost. For the eight-position codon workload those read 32 and 16.

    Both were true, about different things. `preflight.model.load_model` hardcodes
    domain-wall encoding, while `passes.search.compile_spec` searches encodings
    and selected ONE-HOT when the shipped receipt was produced. Same spec, two
    different models, two different spin counts.

    This pins the written description to the receipt on disk, so the two numbers
    the interface shows side by side come from the same compile.
    """
    import json

    receipt_dir = ROOT / "receipts" / record.shelf_id
    if not receipt_dir.is_dir():
        pytest.skip(f"{record.shelf_id} has no packaged receipt here")

    metrics = json.loads((receipt_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_nodes"] == record.receipt_spins, (
        f"{record.shelf_id}: the shelf says {record.receipt_spins} spins but "
        f"the receipt has {metrics['n_nodes']}. The picker shows the receipt's "
        f"number beside the name and the description's number underneath it.")

    passes_path = receipt_dir / "passes.json"
    if passes_path.is_file():
        passes = json.loads(passes_path.read_text(encoding="utf-8"))
        selected = [c for c in passes.get("candidates", [])
                    if c.get("state") == "SELECTED"]
        if selected:
            assert selected[0]["encoding"] == record.receipt_encoding, (
                f"{record.shelf_id}: shelf says {record.receipt_encoding!r}, "
                f"receipt was compiled {selected[0]['encoding']!r}")
        mediators = (passes.get("mediation") or {}).get("mediator_count")
        if mediators is not None:
            assert mediators == record.receipt_mediators, (
                f"{record.shelf_id}: shelf says {record.receipt_mediators} "
                f"mediators, receipt has {mediators}")


def test_the_spin_count_beside_the_name_matches_the_written_description():
    """The two places a spin count appears in the picker must agree."""
    by_id = {item["id"]: item for item in examples_shelf()}
    for record in VERIFIED_WORKLOADS:
        item = by_id[record.shelf_id]
        shown = (item.get("receipt") or {}).get("n_nodes")
        if shown is None:
            continue
        assert shown == record.receipt_spins, (
            f"{record.shelf_id}: picker shows {shown} spins beside the name, "
            f"the description is written for {record.receipt_spins}")


def test_the_shelf_reports_how_many_states_a_program_can_reach():
    """R29. The number was always measured and never displayed.

    Every receipt's verification pass records `diversity_reachable`. Without it
    on screen, `prog_ecology_lotka_lite`, whose contract admits 2 of 65,536
    assignments, looked identical to `prog_codon_opt_tiny` at 729.
    """
    by_id = {i["id"]: i for i in examples_shelf()}
    eco = by_id.get("prog_ecology_lotka_lite")
    if eco is None or not eco["packaged"]:
        pytest.skip("ecology receipt not packaged here")
    assert eco["reachable"] == 2, (
        f"expected 2 reachable states, got {eco['reachable']}; either the "
        f"receipt changed or the shelf is reading the wrong field")

    codon = by_id.get("prog_codon_opt_tiny")
    if codon and codon["packaged"]:
        assert codon["reachable"] == 729


def test_a_space_too_large_to_enumerate_says_so_rather_than_reporting_null():
    """The receipt's own refusal is the honest answer and must survive."""
    strings = [i["reachable"] for i in examples_shelf()
               if isinstance(i["reachable"], str)]
    assert strings, "no receipt reported an un-enumerable space; expected several"
    assert all("unavailable" in s or "too large" in s for s in strings), strings
