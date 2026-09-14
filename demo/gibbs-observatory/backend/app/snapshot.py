"""Snapshot metadata helpers — JSON receipt slice only (no sim dumps)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

APP_VERSION = "0.5.0"

STANDING_PROHIBITIONS: list[dict[str, str]] = [
    {
        "id": "no_silicon",
        "text": "No silicon execution claims — software / THRML+JAX only",
    },
    {
        "id": "no_energy",
        "text": "No energy-savings or joule claims",
    },
    {
        "id": "timing",
        "text": "THRML wall-clock ≠ TSU / silicon timing",
    },
    {
        "id": "thermalizers",
        "text": "Never claim “Thermalizers-complete”",
    },
    {
        "id": "beta_fixed",
        "text": "Mediated β is FIXED (not a free slider)",
    },
    {
        "id": "ess",
        "text": "ESS unavailable when the receipt contract is broken",
    },
    {
        "id": "invalid_draws",
        "text": "Invalid draws never rendered as worlds",
    },
    {
        "id": "state_space",
        "text": "No fake full 2^N state space for large models",
    },
]

DEFAULT_CLAIM_BADGES = [
    "software / THRML+JAX",
    "documented Z1 caps (assumed fields marked)",
    "no silicon",
    "β FIXED when mediated",
    "no energy claims",
    "ESS only when honest",
]


def claim_hygiene_payload(
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    badges = list(DEFAULT_CLAIM_BADGES)
    if receipt and isinstance(receipt.get("claim_badges"), list) and receipt["claim_badges"]:
        badges = list(receipt["claim_badges"])
    # Live honesty flags derived from receipt
    live: list[dict[str, Any]] = []
    if receipt:
        live.append(
            {
                "id": "runtime",
                "ok": True,
                "badge": "software / THRML only",
                "detail": receipt.get("label")
                or "JAX/THRML simulation — not Extropic silicon",
            }
        )
        live.append(
            {
                "id": "silicon",
                "ok": True,
                "badge": "no silicon",
                "detail": "No Extropic silicon path in this Observatory build",
            }
        )
        beta_fixed = bool(receipt.get("beta_fixed"))
        live.append(
            {
                "id": "beta",
                "ok": True,
                "badge": "mediated β FIXED" if beta_fixed else "β editable (unmediated)",
                "detail": f"β={receipt.get('beta')} fixed={beta_fixed}",
            }
        )
        ess = (receipt.get("verification") or {}).get("ess") or {}
        ess_ok = bool(ess.get("available"))
        live.append(
            {
                "id": "ess",
                "ok": ess_ok or not ess.get("reason"),
                "badge": "ESS available" if ess_ok else "ESS unavailable when contract broken",
                "detail": ess.get("reason") or ("ESS reported" if ess_ok else "diagnostic only"),
            }
        )
        live.append(
            {
                "id": "energy_claims",
                "ok": True,
                "badge": "no energy claims",
                "detail": "Energy traces are simulation diagnostics, not hardware joules",
            }
        )
    return {
        "version": APP_VERSION,
        "standing_prohibitions": STANDING_PROHIBITIONS,
        "claim_badges": badges,
        "live": live,
        "label": "JAX/THRML simulation — not Extropic silicon",
    }


def build_snapshot_slice(
    receipt: dict[str, Any] | None = None,
    *,
    client: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact JSON for download — never includes sample dumps or full program.json."""
    client = client or {}
    spins = (receipt or {}).get("spins") or {}
    gates = (receipt or {}).get("gates") or []
    gate_summary = [
        {
            "gate": g.get("gate"),
            "passed": g.get("passed"),
            "measured": g.get("measured"),
            "limit": g.get("limit"),
            "assumed": g.get("assumed", False),
        }
        for g in gates
        if isinstance(g, dict)
    ]
    hygiene = claim_hygiene_payload(receipt)
    return {
        "schema": "gibbs-observatory.snapshot.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
        "receipt_id": (receipt or {}).get("id") or client.get("receipt_id"),
        "verdict": (receipt or {}).get("verdict"),
        "encoding": (receipt or {}).get("encoding"),
        "kernel": (receipt or {}).get("kernel"),
        "beta": (receipt or {}).get("beta"),
        "beta_fixed": bool((receipt or {}).get("beta_fixed")),
        "spins": {
            "n_nodes": spins.get("n_nodes"),
            "world": spins.get("world"),
            "mediators": spins.get("mediators"),
        },
        "gates": gate_summary,
        "claim_badges": hygiene["claim_badges"],
        "standing_prohibitions": [p["text"] for p in STANDING_PROHIBITIONS],
        "step": client.get("step"),
        "active_block": client.get("active_block"),
        "view": client.get("view"),
        "png_filename": client.get("png_filename"),
        "label": "JAX/THRML simulation — not Extropic silicon",
        "notes": [
            "PNG is client-captured from the main stage canvas/view.",
            "This JSON is a receipt slice only — no sim sample dumps.",
            "Do not commit snapshot downloads into git receipts/.",
        ],
    }
