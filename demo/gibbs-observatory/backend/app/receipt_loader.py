"""Load and normalize curated compile receipts for Gibbs Observatory.

Does not invent fields: missing keys are omitted or marked unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default curated shelf (relative to repo root when running from box)
DEFAULT_RECEIPTS_ROOT = Path(__file__).resolve().parents[2] / "receipts"

_OPTIONAL_FILES = (
    "passes.json",
    "program.json",
    "gates.json",
    "formulation.json",
    "workload.json",
    "metrics.json",
    "simulation.json",
    "verification.json",
    "regime.json",
    "target.json",
    "environment.json",
    "cost.json",
    "candidates.json",
    "energy.json",
    "sample.json",
)


def receipts_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_RECEIPTS_ROOT


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None



def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def list_receipts(root: Path | str | None = None) -> list[dict[str, Any]]:
    """List curated receipt directories under receipts/."""
    base = receipts_root(root)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "passes.json").is_file() and not (child / "program.json").is_file():
            continue
        passes = _read_json(child / "passes.json") or {}
        metrics = _read_json(child / "metrics.json") or {}
        program = _read_json(child / "program.json") or {}
        encoding = _selected_encoding(passes)
        mediation = passes.get("mediation") if isinstance(passes, dict) else None
        med = mediation if isinstance(mediation, dict) else {}
        n_nodes = None
        if isinstance(program, dict) and isinstance(program.get("nodes"), list):
            n_nodes = len(program["nodes"])
        elif isinstance(metrics, dict) and "n_nodes" in metrics:
            n_nodes = metrics.get("n_nodes")
        out.append(
            {
                "id": child.name,
                "path": str(child),
                "verdict": passes.get("verdict") if isinstance(passes, dict) else None,
                "encoding": encoding,
                "n_nodes": n_nodes,
                "mediator_count": med.get("mediator_count"),
                "bipartite": metrics.get("bipartite") if isinstance(metrics, dict) else None,
            }
        )
    return out


def _selected_encoding(passes: dict[str, Any] | None) -> str | None:
    if not isinstance(passes, dict):
        return None
    for cand in passes.get("candidates") or []:
        if isinstance(cand, dict) and cand.get("state") == "SELECTED":
            enc = cand.get("encoding")
            return str(enc) if enc is not None else None
    return None


def _normalize_positions(coords: dict[str, Any] | list | None, n: int) -> list[list[float]]:
    """Normalize placement coords to [0,1]^2. Missing → unavailable layout."""
    if coords is None or n <= 0:
        return []
    raw: list[tuple[float, float] | None] = [None] * n
    if isinstance(coords, dict):
        for k, v in coords.items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n and isinstance(v, (list, tuple)) and len(v) >= 2:
                raw[i] = (float(v[0]), float(v[1]))
    elif isinstance(coords, list):
        for i, v in enumerate(coords):
            if i >= n:
                break
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                raw[i] = (float(v[0]), float(v[1]))
    present = [p for p in raw if p is not None]
    if not present:
        return []
    xs = [p[0] for p in present]
    ys = [p[1] for p in present]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    out: list[list[float]] = []
    for p in raw:
        if p is None:
            out.append([0.5, 0.5])
        else:
            out.append([(p[0] - min_x) / span_x, (p[1] - min_y) / span_y])
    return out


def _gate_table(gates: Any) -> list[dict[str, Any]]:
    if not isinstance(gates, list):
        return []
    table: list[dict[str, Any]] = []
    for g in gates:
        if not isinstance(g, dict):
            continue
        table.append(
            {
                "gate": g.get("gate"),
                "passed": g.get("passed"),
                "measured": g.get("measured"),
                "limit": g.get("limit"),
                "assumed": g.get("assumed"),
                "downgraded": g.get("downgraded"),
            }
        )
    return table


def program_drives_thrml(program: dict[str, Any] | None) -> bool:
    """True when program.json has usable spins + edges for Ising sampling."""
    if not isinstance(program, dict):
        return False
    nodes = program.get("nodes")
    edges = program.get("edges")
    weights = program.get("weights")
    biases = program.get("biases")
    if not isinstance(nodes, list) or len(nodes) < 2:
        return False
    if not isinstance(edges, list) or len(edges) < 1:
        return False
    if not isinstance(weights, list) or len(weights) != len(edges):
        return False
    if not isinstance(biases, list) or len(biases) != len(nodes):
        return False
    # edges must be index pairs
    n = len(nodes)
    for e in edges[: min(8, len(edges))]:
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            return False
        i, j = int(e[0]), int(e[1])
        if i < 0 or j < 0 or i >= n or j >= n:
            return False
    return True


def load_receipt(receipt_id: str, root: Path | str | None = None) -> dict[str, Any]:
    """Load a receipt directory into a normalized inspect payload."""
    base = receipts_root(root)
    path = (base / receipt_id).resolve()
    if not str(path).startswith(str(base.resolve())):
        raise FileNotFoundError(f"receipt id escapes root: {receipt_id}")
    if not path.is_dir():
        raise FileNotFoundError(f"receipt not found: {receipt_id}")

    files: dict[str, Any] = {}
    available: list[str] = []
    unavailable: list[str] = []
    for name in _OPTIONAL_FILES:
        data = _read_json(path / name)
        if data is None:
            unavailable.append(name)
        else:
            files[name.replace(".json", "")] = data
            available.append(name)

    passes = files.get("passes") if isinstance(files.get("passes"), dict) else {}
    program = files.get("program") if isinstance(files.get("program"), dict) else {}
    metrics = files.get("metrics") if isinstance(files.get("metrics"), dict) else {}
    workload = files.get("workload") if isinstance(files.get("workload"), dict) else {}
    target = files.get("target") if isinstance(files.get("target"), dict) else {}
    simulation = files.get("simulation") if isinstance(files.get("simulation"), dict) else {}
    verification = files.get("verification") if isinstance(files.get("verification"), dict) else {}
    regime = files.get("regime") if isinstance(files.get("regime"), dict) else {}
    gates_raw = files.get("gates")

    mediation_raw = passes.get("mediation") if passes else None
    mediation = mediation_raw if isinstance(mediation_raw, dict) else {}

    encoding = _selected_encoding(passes)
    beta = program.get("beta") if program else None
    if beta is None and mediation.get("beta_used") is not None:
        beta = mediation.get("beta_used")
    # Mediated models: β FIXED (Lattice rule)
    beta_fixed = bool(mediation) and mediation.get("mediator_count", 0) not in (None, 0)
    if not beta_fixed and mediation.get("beta_used") is not None:
        # still treat as fixed when mediation pass recorded beta_used
        beta_fixed = True

    kernel = None
    if isinstance(program.get("schedule"), str):
        kernel = program["schedule"]
    elif isinstance(target.get("schedule"), dict) and target["schedule"].get("value"):
        kernel = target["schedule"]["value"]

    nodes = program.get("nodes") if isinstance(program.get("nodes"), list) else []
    edges_raw = program.get("edges") if isinstance(program.get("edges"), list) else []
    weights = program.get("weights") if isinstance(program.get("weights"), list) else []
    biases = program.get("biases") if isinstance(program.get("biases"), list) else []
    blocks = program.get("blocks") if isinstance(program.get("blocks"), list) else []
    mediator_nodes = (
        list(program.get("mediator_nodes") or [])
        if isinstance(program.get("mediator_nodes"), list)
        else []
    )
    mediator_set = set(int(i) for i in mediator_nodes)
    n_spins = len(nodes)
    world_idx = [i for i in range(n_spins) if i not in mediator_set]
    mediator_idx = sorted(mediator_set)

    edges: list[list[int]] = []
    for e in edges_raw:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            edges.append([int(e[0]), int(e[1])])

    placement = program.get("placement") if isinstance(program.get("placement"), dict) else {}
    positions = _normalize_positions(placement.get("coords"), n_spins)

    color0: list[int] = []
    color1: list[int] = []
    if len(blocks) >= 2 and isinstance(blocks[0], list) and isinstance(blocks[1], list):
        color0 = [int(i) for i in blocks[0]]
        color1 = [int(i) for i in blocks[1]]
    elif n_spins:
        # fallback bipartite guess unavailable — leave empty rather than invent
        color0, color1 = [], []

    thrml_ok = program_drives_thrml(program)

    # Frontier / capacity from gates + metrics (only present fields)
    frontier: dict[str, Any] = {}
    if metrics:
        for k in ("max_degree", "max_abs_J", "max_abs_b", "bipartite", "colour_blocks", "n_nodes", "n_edges"):
            if k in metrics:
                frontier[k] = metrics[k]
    if target:
        for key, out_key in (
            ("degree", "degree_limit"),
            ("max_abs_coupling", "coupling_cap"),
            ("max_abs_bias", "field_cap"),
            ("node_budget", "node_budget"),
        ):
            t = target.get(key)
            if isinstance(t, dict) and "value" in t:
                frontier[out_key] = t["value"]

    # Logical vs physical (Fabric Tax) — only from real fields
    connectivity: dict[str, Any] = {
        "logical": {},
        "physical": {},
        "notes": [],
    }
    if workload:
        if "variables" in workload:
            connectivity["logical"]["variables"] = workload["variables"]
        if "logical_interactions" in workload:
            connectivity["logical"]["interactions"] = workload["logical_interactions"]
        if "state_cardinality" in workload:
            connectivity["logical"]["state_cardinality"] = workload["state_cardinality"]
    if metrics:
        if "n_nodes" in metrics:
            connectivity["physical"]["n_nodes"] = metrics["n_nodes"]
        if "n_edges" in metrics:
            connectivity["physical"]["n_edges"] = metrics["n_edges"]
        if "bipartite" in metrics:
            connectivity["physical"]["bipartite"] = metrics["bipartite"]
        if "mediators" in metrics:
            connectivity["physical"]["mediators_field"] = metrics["mediators"]
    if n_spins:
        connectivity["physical"]["program_n_nodes"] = n_spins
        connectivity["physical"]["program_n_edges"] = len(edges)
        connectivity["physical"]["world_spins"] = len(world_idx)
        connectivity["physical"]["mediator_spins"] = len(mediator_idx)
    if mediation:
        if "bipartite_after" in mediation:
            connectivity["physical"]["bipartite_after"] = mediation["bipartite_after"]
        if "partition_method" in mediation:
            connectivity["physical"]["partition_method"] = mediation["partition_method"]
        if "mediator_count" in mediation:
            connectivity["physical"]["mediator_count"] = mediation["mediator_count"]
    if not connectivity["logical"]:
        connectivity["notes"].append(
            "Logical graph edge list not present in receipt; "
            "workload counts only (if any)."
        )
    if "mediators" in metrics and metrics.get("mediators") == 0 and mediator_idx:
        connectivity["notes"].append(
            "metrics.json mediators=0 but program.json lists mediator_nodes; "
            "using program.json mediator_nodes for spin counts."
        )

    # ESS honesty from verification / regime
    ess = verification.get("ess") if verification else None
    ess_honest: dict[str, Any] = {"available": False, "value": None, "reason": None}
    if isinstance(ess, (int, float)):
        ess_honest = {"available": True, "value": ess, "reason": None}
    elif isinstance(ess, str):
        ess_honest = {"available": False, "value": None, "reason": ess}
    elif regime.get("mixing_indicator"):
        ess_honest = {
            "available": False,
            "value": None,
            "reason": regime.get("mixing_indicator"),
        }

    sampling: dict[str, Any] = {
        "thrml_ready": thrml_ok,
        "fallback": None if thrml_ok else "lattice2d",
        "banner": None
        if thrml_ok
        else "sampling fallback: preset (receipt graph not yet wired)",
        "beta": beta,
        "beta_fixed": beta_fixed,
        "seed": None,
        "n_chains": None,
    }
    if isinstance(simulation.get("params"), dict):
        sp = simulation["params"]
        sampling["seed"] = sp.get("seed")
        sampling["n_chains"] = sp.get("n_chains")
        if sampling["beta"] is None and "beta" in sp:
            sampling["beta"] = sp["beta"]

    fabric_tax = derive_fabric_tax(edges, mediator_idx, nodes if nodes else None)
    schedule = {
        "kernel": kernel,
        "n_colours": 2 if (color0 or color1) else (len(blocks) if blocks else None),
        "blocks": [
            {"colour": 0, "size": len(color0), "indices": color0},
            {"colour": 1, "size": len(color1), "indices": color1},
        ]
        if color0 or color1
        else None,
        "notes": [
            "Chromatic block-Gibbs: alternate colour-0 and colour-1 updates.",
            "active_block from the live WS stream highlights which colour updates.",
        ]
        if color0 or color1
        else ["No chromatic blocks in program.json — schedule timeline unavailable."],
    }

    payload: dict[str, Any] = {
        "id": receipt_id,
        "path": str(path),
        "verdict": passes.get("verdict"),
        "ideal_passed": passes.get("ideal_passed"),
        "hardware_evaluated": passes.get("hardware_evaluated"),
        "encoding": encoding,
        "beta": beta,
        "beta_fixed": beta_fixed,
        "kernel": kernel,
        "gates": _gate_table(gates_raw),
        "spins": {
            "n_nodes": n_spins or metrics.get("n_nodes"),
            "world": len(world_idx) if n_spins else None,
            "mediators": len(mediator_idx) if n_spins else mediation.get("mediator_count"),
            "world_idx": world_idx,
            "mediator_idx": mediator_idx,
            "node_names": nodes if nodes else None,
        },
        "edges": {
            "count": len(edges) if edges else metrics.get("n_edges"),
            "pairs": edges if edges else None,
            "weights": [float(w) for w in weights] if weights and len(weights) == len(edges) else None,
        },
        "biases": [float(b) for b in biases] if biases and n_spins and len(biases) == n_spins else None,
        "blocks": {"color0": color0, "color1": color1} if color0 or color1 else None,
        "positions": positions if positions else None,
        "frontier": frontier or None,
                "connectivity": connectivity,
        "mediation": mediation or None,
        "fabric_tax": fabric_tax,
        "schedule": schedule,
        "workload": workload or None,
        "metrics": metrics or None,
        "verification": {
            "task_validity": verification.get("task_validity"),
            "codeword_violation_rate": verification.get("codeword_violation_rate"),
            "ess": ess_honest,
        }
        if verification
        else None,
        "regime": {
            "regime": regime.get("regime"),
            "coupling_utilisation": regime.get("coupling_utilisation"),
            "precision_headroom": regime.get("precision_headroom"),
        }
        if regime
        else None,
        "target": {
            "name": target.get("name"),
            "bipartite": target.get("bipartite", {}).get("value")
            if isinstance(target.get("bipartite"), dict)
            else None,
        }
        if target
        else None,
        "formulation_summary": _formulation_summary(files.get("formulation")),
        "sampling": sampling,
        "files_available": available,
        "files_unavailable": unavailable,
        "label": "JAX/THRML simulation — not Extropic silicon",
        "claim_badges": [
            "software / THRML+JAX",
            "documented Z1 caps (assumed fields marked)",
            "no silicon",
            "β FIXED when mediated",
            "no energy claims",
            "ESS only when honest",
        ],
        "spec_yaml": _read_text(path / "spec.yaml"),
    }
    residuals = derive_residuals(
        workload=workload or None,
        physical=connectivity.get("physical") if connectivity else None,
        frontier=frontier or None,
        gates=payload["gates"],
        verification=verification or None,
        mediation=mediation or None,
    )
    payload["residuals"] = residuals

    return payload


def _formulation_summary(formulation: Any) -> list[dict[str, Any]] | None:
    if not isinstance(formulation, list):
        return None
    out = []
    for item in formulation:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "construct": item.get("construct"),
                "ir_shape": item.get("ir_shape"),
                "count": item.get("count"),
                "scope": item.get("scope"),
            }
        )
    return out or None


def receipt_to_graph_arrays(receipt_id: str, root: Path | str | None = None) -> dict[str, Any]:
    """Extract THRML-ready arrays from a receipt (raises if not thrml-ready)."""
    payload = load_receipt(receipt_id, root=root)
    if not payload["sampling"]["thrml_ready"]:
        raise ValueError(f"receipt {receipt_id} cannot drive THRML yet")
    edges = payload["edges"]["pairs"]
    weights = payload["edges"]["weights"]
    biases = payload["biases"]
    assert edges is not None and weights is not None and biases is not None
    blocks = payload["blocks"] or {"color0": [], "color1": []}
    return {
        "n_nodes": int(payload["spins"]["n_nodes"]),
        "edges": [(int(a), int(b)) for a, b in edges],
        "weights": [float(w) for w in weights],
        "biases": [float(b) for b in biases],
        "beta": float(payload["beta"]) if payload["beta"] is not None else 1.0,
        "beta_fixed": bool(payload["beta_fixed"]),
        "color0": list(blocks.get("color0") or []),
        "color1": list(blocks.get("color1") or []),
        "world_idx": list(payload["spins"]["world_idx"]),
        "mediator_idx": list(payload["spins"]["mediator_idx"]),
        "positions": payload["positions"] or [],
        "node_names": payload["spins"]["node_names"] or [],
        "kernel": payload["kernel"],
        "receipt_id": receipt_id,
    }


# --- Phase 2 helpers (derived metrics labeled as derived; no invented receipt fields) ---

_SHELF_CATALOG: list[dict[str, Any]] = [
    {
        "id": "small",
        "title": "Toy grid (small)",
        "source": "local compile / Lattice receipts/small",
        "kind": "curated",
        "extropic": False,
        "notes": "Default fast receipt; mediated domain-wall toy.",
    },
    {
        "id": "elev_band",
        "title": "Elevation band",
        "source": "Lattice demo receipts/elev_band",
        "kind": "curated",
        "extropic": False,
        "notes": "COMPILED bipartite band; no mediators; THRML-ready.",
    },
    {
        "id": "prog_alloy_ordering_8x8",
        "title": "Ordering alloy 8×8 (Distribution Lab)",
        "source": "programs/alloy_ordering_8x8.yaml → receipts/prog_alloy_ordering_8x8",
        "kind": "curated",
        "extropic": False,
        "status": "ready",
        "packaged": True,
        "notes": "Original Observatory gift — Bragg–Williams / Ising ordering alloy. Not Extropic codon.",
    },
    {
        "id": "prog_ebm_bars_stripes",
        "title": "EBM bars-and-stripes 6×6 (first AI demo)",
        "source": "programs/ebm_bars_stripes.yaml → receipts/prog_ebm_bars_stripes",
        "kind": "curated",
        "extropic": False,
        "status": "ready",
        "packaged": True,
        "notes": "Trained pairwise Ising on classic bars-and-stripes. Train→compile→EBM Lab. THRML/JAX SIM — not silicon.",
    },
    {
        "id": "prog_codon_opt_tiny",
        "title": "codon_opt tiny (SEQ)",
        "source": "programs/codon_opt_tiny.yaml → receipts/prog_codon_opt_tiny",
        "kind": "curated",
        "extropic": True,
        "notes": "Extropic/THRML codon class; Z1 COMPILED; notepad YAML in programs/.",
    },
    {
        "id": "prog_number_partition",
        "title": "Number partition",
        "source": "programs/number_partition.yaml → receipts/prog_number_partition",
        "kind": "curated",
        "extropic": False,
        "notes": "Classic Ising partition; n=12 → 3D state space OK.",
    },
    {
        "id": "prog_ecology_lotka_lite",
        "title": "Ecology competitive 4×4",
        "source": "programs/ecology_lotka_lite.yaml → receipts/prog_ecology_lotka_lite",
        "kind": "curated",
        "extropic": True,
        "notes": "Discrete competitive exclusion; Torx LV continuous still later.",
    },
    {
        "id": "prog_jobshop_tiny",
        "title": "Job-shop lite 3×2",
        "source": "programs/jobshop_tiny.yaml → receipts/prog_jobshop_tiny",
        "kind": "curated",
        "extropic": False,
        "notes": "Manufacturing assignment + contention; Z1 COMPILED.",
    },
    {
        "id": "prog_market_binary_factors",
        "title": "Banded market factors",
        "source": "programs/market_binary_factors.yaml → receipts/prog_market_binary_factors",
        "kind": "curated",
        "extropic": False,
        "notes": "Sparse factor graph (not dense K20); degree ≪ 16.",
    },
    {
        "id": "prog_seq_design_longer",
        "title": "SEQ design 8 positions",
        "source": "programs/seq_design_longer.yaml → receipts/prog_seq_design_longer",
        "kind": "curated",
        "extropic": True,
        "notes": "codon_opt scale-up; 8 AA positions; Z1 COMPILED.",
    },
    {
        "id": "prog_knapsack_tiny",
        "title": "Tiny knapsack",
        "source": "programs/knapsack_tiny.yaml → receipts/prog_knapsack_tiny",
        "kind": "curated",
        "extropic": False,
        "notes": "0/1 knapsack soft capacity; resource allocation.",
    },
    {
        "id": "prog_sat_3tiny",
        "title": "Tiny 3-SAT (Rosenberg)",
        "source": "programs/sat_3tiny.yaml → receipts/prog_sat_3tiny",
        "kind": "curated",
        "extropic": False,
        "notes": "3-SAT via pairwise + auxiliary QUBO reduction.",
    },
    {
        "id": "prog_maxcut_cycle7",
        "title": "Max-Cut C7",
        "source": "programs/maxcut_cycle7.yaml → receipts/prog_maxcut_cycle7",
        "kind": "curated",
        "extropic": False,
        "notes": "Odd-cycle Max-Cut; one mediator.",
    },
    {
        "id": "prog_roster_shift_conflicts",
        "title": "Roster shift conflicts",
        "source": "programs/roster_shift_conflicts.yaml → receipts/prog_roster_shift_conflicts",
        "kind": "curated",
        "extropic": False,
        "notes": "Smallest mediated scheduling demo.",
    },
    {
        "id": "prog_placement_8x8_exclusion",
        "title": "Placement 8×8 exclusion",
        "source": "programs/placement_8x8_exclusion.yaml → receipts/prog_placement_8x8_exclusion",
        "kind": "curated",
        "extropic": False,
        "notes": "Hard-core packing; n=64 (2D state space, not 3D).",
    },
    {
        "id": "codon_opt",
        "title": "codon_opt (legacy stub id)",
        "source": "use prog_codon_opt_tiny instead",
        "kind": "stub",
        "extropic": True,
        "notes": "Legacy shelf id — real receipt is prog_codon_opt_tiny.",
    },
]


def derive_fabric_tax(
    edges: list[list[int]],
    mediator_nodes: list[int],
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Derive mediator callouts from physical edges (degree-2 mediators).

    When each mediator connects exactly two non-mediator (world) spins, treat
    that pair as a mediated logical interaction. Labeled derived — not a
    receipt field.
    """
    med_set = {int(i) for i in mediator_nodes}
    if not med_set or not edges:
        return {
            "derived": True,
            "method": "mediator_world_neighbors",
            "pair_count": 0,
            "pairs": [],
            "by_edge_key": {},
            "notes": [
                "No mediators in program.json — Fabric Tax click-to-mediator unavailable."
                if not med_set
                else "No edges to derive mediation pairs from."
            ],
        }

    from collections import defaultdict

    neighbors: dict[int, list[int]] = defaultdict(list)
    for e in edges:
        if len(e) < 2:
            continue
        a, b = int(e[0]), int(e[1])
        if a in med_set and b not in med_set:
            neighbors[a].append(b)
        elif b in med_set and a not in med_set:
            neighbors[b].append(a)

    pairs: list[dict[str, Any]] = []
    by_edge_key: dict[str, dict[str, Any]] = {}
    skipped = 0
    for m in sorted(med_set):
        world = sorted(set(neighbors.get(m, [])))
        if len(world) != 2:
            skipped += 1
            continue
        u, v = world[0], world[1]
        key = f"{min(u, v)}-{max(u, v)}"
        entry = {
            "world_u": u,
            "world_v": v,
            "mediator": m,
            "world_u_name": (node_names[u] if node_names and u < len(node_names) else None),
            "world_v_name": (node_names[v] if node_names and v < len(node_names) else None),
            "mediator_name": (
                node_names[m] if node_names and m < len(node_names) else None
            ),
        }
        pairs.append(entry)
        by_edge_key[key] = entry

    notes = [
        "Derived from program.json mediator_nodes + edges (mediators with "
        "exactly two world neighbors). Not a stored logical-edge list."
    ]
    if skipped:
        notes.append(
            f"{skipped} mediator(s) skipped (neighbor count ≠ 2); "
            "no invented callout for those."
        )
    return {
        "derived": True,
        "method": "mediator_world_neighbors",
        "pair_count": len(pairs),
        "pairs": pairs,
        "by_edge_key": by_edge_key,
        "notes": notes,
    }


def derive_residuals(
    *,
    workload: dict[str, Any] | None,
    physical: dict[str, Any] | None,
    frontier: dict[str, Any] | None,
    gates: list[dict[str, Any]] | None,
    verification: dict[str, Any] | None,
    mediation: dict[str, Any] | None,
) -> dict[str, Any]:
    """v0 residual / headroom metrics from existing receipt fields only.

    Anything computed here is marked derived=True. Missing sources → honest gaps.
    """
    workload = workload or {}
    physical = physical or {}
    frontier = frontier or {}
    gates = gates or []
    verification = verification or {}
    mediation = mediation or {}

    bars: list[dict[str, Any]] = []
    heatmap: list[dict[str, Any]] = []
    unavailable: list[str] = []
    notes: list[str] = []

    # Connectivity residual: logical interactions vs physical edges (counts only)
    log_i = workload.get("logical_interactions")
    phys_e = physical.get("program_n_edges") or physical.get("n_edges")
    log_v = workload.get("variables")
    phys_n = physical.get("program_n_nodes") or physical.get("n_nodes")
    med_n = physical.get("mediator_spins")
    if med_n is None:
        med_n = physical.get("mediator_count") or mediation.get("mediator_count")

    if isinstance(log_i, (int, float)) and isinstance(phys_e, (int, float)):
        delta = float(phys_e) - float(log_i)
        bars.append(
            {
                "id": "edge_count_delta",
                "label": "physical_edges − logical_interactions",
                "value": delta,
                "derived": True,
                "sources": ["workload.logical_interactions", "program/metrics edges"],
            }
        )
        heatmap.append(
            {
                "row": "connectivity",
                "col": "edge_delta",
                "value": delta,
                "derived": True,
            }
        )
    else:
        unavailable.append(
            "Connectivity residual (edge delta): need workload.logical_interactions "
            "and physical edge count."
        )

    if isinstance(log_v, (int, float)) and isinstance(phys_n, (int, float)):
        spin_delta = float(phys_n) - float(log_v)
        bars.append(
            {
                "id": "spin_count_delta",
                "label": "physical_nodes − logical_variables",
                "value": spin_delta,
                "derived": True,
                "sources": ["workload.variables", "program/metrics n_nodes"],
            }
        )
        heatmap.append(
            {
                "row": "connectivity",
                "col": "spin_delta",
                "value": spin_delta,
                "derived": True,
            }
        )
        if isinstance(med_n, (int, float)):
            bars.append(
                {
                    "id": "mediator_vs_spin_delta",
                    "label": "mediators (should ≈ spin_delta when DW-mediated)",
                    "value": float(med_n),
                    "derived": True,
                    "sources": ["program.mediator_nodes or passes.mediation"],
                }
            )

    # Degree / coupling / field headroom from gates (limit − measured)
    for g in gates:
        if not isinstance(g, dict):
            continue
        name = g.get("gate")
        measured = g.get("measured")
        limit = g.get("limit")
        if name in ("degree", "coupling_cap", "field_cap", "node_budget") and isinstance(
            measured, (int, float)
        ) and isinstance(limit, (int, float)):
            headroom = float(limit) - float(measured)
            util = float(measured) / float(limit) if float(limit) != 0 else None
            bars.append(
                {
                    "id": f"headroom_{name}",
                    "label": f"{name} headroom (limit − measured)",
                    "value": headroom,
                    "utilisation": util,
                    "derived": True,
                    "sources": [f"gates.json:{name}"],
                    "assumed": bool(g.get("assumed")),
                }
            )
            heatmap.append(
                {
                    "row": "capacity",
                    "col": str(name),
                    "value": headroom,
                    "utilisation": util,
                    "derived": True,
                }
            )

    # Frontier utilisation if present without inventing
    if isinstance(frontier.get("max_degree"), (int, float)) and isinstance(
        frontier.get("degree_limit"), (int, float)
    ):
        # already covered by gates usually; skip duplicate unless gate missing
        pass

    # Verification transport / TV — surface as unavailable strings, never invent numbers
    for key in ("energy_tv", "execution_tv", "cross_check_tv", "execution_noise_floor"):
        val = verification.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            bars.append(
                {
                    "id": key,
                    "label": key,
                    "value": float(val),
                    "derived": False,
                    "sources": [f"verification.json:{key}"],
                }
            )
        elif isinstance(val, str):
            unavailable.append(f"{key}: {val}")

    if not bars and not unavailable:
        notes.append(
            "No residual sources in receipt (workload / gates / verification). "
            "Empty state is intentional."
        )
    else:
        notes.append(
            "Bars marked derived=true are computed in Observatory from existing "
            "receipt fields; they are not stored residual matrices."
        )

    return {
        "bars": bars,
        "heatmap": heatmap,
        "unavailable": unavailable,
        "notes": notes,
        "has_receipt_residual_matrix": False,
    }


def examples_shelf(root: Path | str | None = None) -> list[dict[str, Any]]:
    """Curated Extropic / Lattice examples shelf (includes stubs)."""
    base = receipts_root(root)
    on_disk = {r["id"]: r for r in list_receipts(root)}
    out: list[dict[str, Any]] = []
    for entry in _SHELF_CATALOG:
        eid = entry["id"]
        packaged = eid in on_disk
        # codon_opt dir may exist with README only — still stub
        stub = entry.get("kind") == "stub" or (
            not packaged
            and not (base / eid / "program.json").is_file()
            and not (base / eid / "passes.json").is_file()
        )
        item = {
            **entry,
            "packaged": packaged and not stub,
            "status": "stub" if stub else ("ready" if packaged else "missing"),
            "message": (
                "receipt not packaged yet"
                if stub
                else None
            ),
            "receipt": on_disk.get(eid) if packaged and not stub else None,
        }
        out.append(item)
    # Also surface any on-disk receipts not in catalog
    catalog_ids = {e["id"] for e in _SHELF_CATALOG}
    for rid, meta in on_disk.items():
        if rid in catalog_ids:
            continue
        out.append(
            {
                "id": rid,
                "title": rid,
                "source": meta.get("path"),
                "kind": "local",
                "extropic": False,
                "notes": "On-disk receipt not in curated catalog.",
                "packaged": True,
                "status": "ready",
                "message": None,
                "receipt": meta,
            }
        )
    return out
