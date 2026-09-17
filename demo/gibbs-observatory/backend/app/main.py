"""FastAPI + WebSocket server for Gibbs Observatory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .program_service import (
    preflight_edges,
    ProgramServiceError,
    apply_program,
    compile_program,
    gate_preview,
    preflight_program,
    read_spec_yaml,
    tsu_status,
)
from .receipt_loader import examples_shelf, list_receipts, load_receipt
from .spec_formats import DetectionError, load_text
from .program_contract import ConsentRequired, ProgramError, describe_source
from .program_runtime import DEFAULT_SWEEPS, open_session
from .sampler_engine import SamplerConfig, SamplerEngine
from .snapshot import (
    APP_VERSION,
    build_snapshot_slice,
    claim_hygiene_payload,
)
from .ebm_bars_stripes import (
    ARTIFACT_DIR,
    CAPTION as EBM_CAPTION,
    EBM_PROGRAM_NAME,
    EBM_RECEIPT_ID,
    GRID as EBM_GRID,
    HONESTY as EBM_HONESTY,
    N_HIDDEN as EBM_N_HIDDEN,
    N_VISIBLE as EBM_N_VISIBLE,
    decode_ebm_sample,
    load_checkpoint,
    train_and_export,
)

app = FastAPI(
    title="Gibbs Observatory",
    description="Nsight-style compiled-program inspector for THRML block-Gibbs sampling",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared engine for REST; WebSocket sessions get their own engines.
_engine = SamplerEngine(SamplerConfig())


class ConfigBody(BaseModel):
    preset: str = "lattice2d"
    size: int = Field(16, ge=4, le=48)
    degree_cap: int = Field(16, ge=2, le=16)
    beta: float = Field(0.5, ge=0.01, le=5.0)
    J: float = Field(1.0, ge=-3.0, le=3.0)
    h: float = Field(0.0, ge=-2.0, le=2.0)
    warmup: int = Field(50, ge=0, le=2000)
    steps_per_sample: int = Field(2, ge=1, le=32)
    batch_size: int = Field(8, ge=1, le=64)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    clamp: bool = False
    receipt_id: str | None = None


class ParamPatch(BaseModel):
    beta: float | None = None
    J: float | None = None
    h: float | None = None
    clamp: bool | None = None
    warmup: int | None = None
    steps_per_sample: int | None = None
    batch_size: int | None = None
    seed: int | None = None


class EdgesBody(BaseModel):
    """A model given directly as an edge list, for data-driven programs.

    The notepad's YAML describes a problem declaratively -- a grid plus uniform
    rules. That cannot express a model whose biases ARE the input data, like the
    playable level's visibility lattice, where every one of 11,200 biases comes
    from the occupancy in front of the player and changes as they move.
    `load_model` has always accepted an edge list; this exposes it.
    """

    edges_json: str = Field(..., min_length=1)
    allow_assumed: bool = False


class ProgramBody(BaseModel):
    """YAML / tsu_compiler.spec text for a Thermodynamic Program."""

    yaml: str = Field(..., min_length=1)
    allow_assumed: bool = False
    target: str = "z1"
    receipt_id: str = "notepad"


class ApplyBody(BaseModel):
    receipt_id: str = Field(..., min_length=1)


class LoadBody(BaseModel):
    """A file the reader picked or text they pasted.

    A browser file picker and a paste produce the same two things: a name and
    some text. `filename` may be empty for a paste, in which case the format is
    decided by the content alone.
    """

    filename: str = ""
    text: str = ""


class LoadPathBody(BaseModel):
    """A path to a program already on disk. This is a desktop app, and
    "point it at the file I wrote" is what people mean by loading one."""

    path: str = Field(..., min_length=1)


#: Refuse anything larger rather than hanging the editor on it. A thermodynamic
#: program is a declaration; a multi-megabyte one is a data dump that belongs
#: at the edge-list door instead.
MAX_PROGRAM_BYTES = 2 * 1024 * 1024


def _loaded_response(filename: str, text: str) -> dict[str, Any]:
    try:
        loaded = load_text(filename, text)
    except DetectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "format": loaded.detection.format,
        "why": loaded.detection.why,
        "executed": loaded.detection.executed,
        "yaml": loaded.yaml,
        "message": loaded.message,
        "filename": filename,
    }


def _config_from_msg(msg: dict[str, Any]) -> SamplerConfig:
    fields = SamplerConfig.__dataclass_fields__
    kwargs: dict[str, Any] = {}
    for k in fields:
        if k not in msg:
            continue
        # Allow explicit null for receipt_id to clear receipt mode
        if k == "receipt_id" or msg[k] is not None:
            kwargs[k] = msg[k]
    return SamplerConfig(**kwargs)


def _program_http(exc: ProgramServiceError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"message": str(exc), "extra": exc.detail},
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "gibbs-observatory",
        "backend": "thrml+jax",
        "version": APP_VERSION,
        "label": "JAX/THRML simulation, not Extropic silicon",
        "tsu": tsu_status(),
    }


@app.get("/api/presets")
def presets() -> dict[str, Any]:
    return {
        "presets": [
            {
                "id": "lattice2d",
                "name": "2D Ising lattice",
                "desc": "Checkerboard chromatic 2-coloring, nearest-neighbor",
                "default_size": 16,
            },
            {
                "id": "chain1d",
                "name": "1D Ising chain",
                "desc": "Even/odd bipartite blocks",
                "default_size": 32,
            },
            {
                "id": "sparse",
                "name": "Sparse degree-capped",
                "desc": "Bipartite random graph, degree ≤ 16",
                "default_size": 40,
            },
        ]
    }


@app.get("/api/receipts")
def api_list_receipts() -> dict[str, Any]:
    return {"receipts": list_receipts()}


@app.get("/api/receipts/{receipt_id}")
def api_get_receipt(receipt_id: str) -> dict[str, Any]:
    try:
        return load_receipt(receipt_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/receipts/{receipt_id}/spec")
def api_get_receipt_spec(receipt_id: str) -> dict[str, Any]:
    try:
        return read_spec_yaml(receipt_id)
    except ProgramServiceError as exc:
        raise _program_http(exc) from exc


@app.get("/api/examples")
def api_examples() -> dict[str, Any]:
    """Curated Extropic / Lattice examples shelf (includes stubs)."""
    return {"examples": examples_shelf()}


@app.get("/api/program/status")
def api_program_status() -> dict[str, Any]:
    return tsu_status()


@app.post("/api/program/gates")
def api_program_gates(body: ProgramBody) -> dict[str, Any]:
    """The gates alone, cheap enough to run while someone types.

    Deliberately NOT a compile: no placement, no routing, no verdict, and the
    response says so in three separate fields. Calling it one would let a
    progress indicator make a claim the compiler never made.
    """
    try:
        return gate_preview(body.yaml, target=body.target,
                            allow_assumed=body.allow_assumed)
    except ProgramServiceError as exc:
        raise _program_http(exc) from exc


@app.post("/api/program/preflight")
def api_program_preflight(body: ProgramBody) -> dict[str, Any]:
    try:
        return preflight_program(body.yaml, allow_assumed=body.allow_assumed)
    except ProgramServiceError as exc:
        raise _program_http(exc) from exc


@app.post("/api/program/preflight-edges")
def api_program_preflight_edges(body: EdgesBody) -> dict[str, Any]:
    try:
        return preflight_edges(body.edges_json, allow_assumed=body.allow_assumed)
    except ProgramServiceError as exc:
        raise _program_http(exc) from exc


@app.post("/api/program/compile")
def api_program_compile(body: ProgramBody) -> dict[str, Any]:
    try:
        return compile_program(
            body.yaml,
            allow_assumed=body.allow_assumed,
            target=body.target,
            receipt_id=body.receipt_id or "notepad",
        )
    except ProgramServiceError as exc:
        raise _program_http(exc) from exc


class ProgramOpenBody(BaseModel):
    """Start a running program. `consent` is the user's explicit yes to
    executing the file; it is never inferred and never defaulted true."""

    path: str = Field(..., min_length=1)
    consent: bool = False
    sweeps: int = Field(DEFAULT_SWEEPS, ge=1, le=256)
    seed: int = Field(0, ge=0, le=2**31 - 1)


class ProgramStepBody(BaseModel):
    session: str = Field(..., min_length=1)
    #: New input for declared ports. Biases only; a graph change is refused.
    inputs: dict[str, Any] = Field(default_factory=dict)
    decode: bool = True


#: Running programs, by id. One process, one user, so a dict is the right size.
_SESSIONS: dict[str, Any] = {}


def _resolve_program(raw: str) -> Path:
    """A program by name, or by path.

    The frontend asks for "visibility_world.py" and should not have to know
    where the application keeps its programs -- which differs between a repo
    checkout and a frozen bundle. A name with no directory part is resolved
    against the shipped programs directory; anything else is taken as a path,
    so a reader can still point at a file of their own.
    """
    candidate = Path(raw).expanduser()
    if candidate.parent != Path("."):
        return candidate
    from .program_service import resource_programs_root
    return resource_programs_root() / candidate.name


@app.post("/api/program/inspect")
def api_program_inspect(body: LoadPathBody) -> dict[str, Any]:
    """What kind of file is this, decided WITHOUT running it.

    This is what lets the Workbench say "this contains code and will execute"
    before asking whether to. Deciding by import would mean running the file to
    find out whether running it was acceptable.
    """
    path = _resolve_program(body.path)
    if path.is_dir():
        raise HTTPException(status_code=400, detail=f"{path} is a directory")
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"no file at {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400,
                            detail=f"could not read {path.name}: {exc}") from exc

    described = describe_source(text) if path.suffix.lower() == ".py" else None
    return {
        "filename": path.name,
        "path": str(path),
        "kind": described.kind if described else "data",
        "why": described.why if described else "not a .py, so it is read as data",
        "executed": False,
        "needs_consent": bool(described and described.kind == "program"),
    }


def _world_of(session: Any) -> list[list[int]] | None:
    """The program's world map as 0/1 rows, if it has one."""
    import numpy as _np
    world = getattr(session.program.module, "WORLD", None)
    if world is None:
        return None
    arr = _np.asarray(world)
    if arr.ndim != 2:
        return None
    return arr.astype(int).tolist()


@app.post("/api/program/open")
def api_program_open(body: ProgramOpenBody) -> dict[str, Any]:
    """Load a program and start a session. Executes the file, with consent."""
    import uuid
    try:
        session = open_session(_resolve_program(body.path), {},
                               consent=body.consent, sweeps=body.sweeps,
                               seed=body.seed)
    except ConsentRequired as exc:
        # 403, not 400: the request is well formed and was refused on purpose.
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ProgramError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = session
    return {
        "session": sid,
        "name": session.program.name,
        "decoder": session.program.decoder,
        "n_spins": session.sampler.n_spins,
        "n_couplings": len(session.model.edges),
        "bipartite": bool(session.report.bipartite),
        "ports": [
            {"name": p.name, "shape": list(p.shape), "mode": p.mode, "doc": p.doc}
            for p in session.program.ports
        ],
        "sweeps": body.sweeps,
        # A program may expose a world map. The viewport needs it to stop the
        # player walking through walls: movement has to be resolved on the
        # frame the key is held, and a round trip per keystroke would make the
        # controls lag behind the picture. The world is level data, not a
        # secret, and the program remains the one place it is defined.
        "world": _world_of(session),
        "label": "JAX/THRML simulation, not Extropic silicon",
    }


@app.post("/api/program/step")
def api_program_step(body: ProgramStepBody) -> dict[str, Any]:
    session = _SESSIONS.get(body.session)
    if session is None:
        raise HTTPException(status_code=404,
                            detail=f"no running program {body.session!r}")
    try:
        frame = session.step(body.inputs or None)
        decoded = session.decode(frame) if body.decode else None
    except ProgramError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    import numpy as _np
    return {
        "frames": frame.frames,
        "n_spins": frame.n_spins,
        # traces > 1 means the sampler recompiled, which costs ~300ms a frame
        # and is otherwise invisible (R32). Reported so it cannot hide.
        "traces": frame.traces,
        "spins": frame.spins[0].astype(int).tolist(),
        "decoded": (_np.asarray(decoded).tolist() if decoded is not None else None),
    }


@app.post("/api/program/close")
def api_program_close(body: ProgramStepBody) -> dict[str, Any]:
    existed = _SESSIONS.pop(body.session, None) is not None
    return {"closed": existed}


@app.post("/api/program/load")
def api_program_load(body: LoadBody) -> dict[str, Any]:
    """Read a program from a picked file or a paste."""
    return _loaded_response(body.filename, body.text)


@app.post("/api/program/load-path")
def api_program_load_path(body: LoadPathBody) -> dict[str, Any]:
    """Read a program from a path on disk.

    Every refusal below names what is actually wrong, because "could not load
    file" sends the reader looking in the wrong place.
    """
    path = Path(body.path).expanduser()
    if path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"{path} is a directory. Point at the program file inside it.")
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"no file at {path}")

    size = path.stat().st_size
    if size > MAX_PROGRAM_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"{path.name} is {size / 1024 / 1024:.1f} MB, over the "
                   f"{MAX_PROGRAM_BYTES / 1024 / 1024:.0f} MB limit for a "
                   f"program. A model this size is data rather than a "
                   f"declaration; load it through the edge-list door instead.")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{path.name} is not UTF-8 text, so it is not a program the "
                   f"Workbench can read.") from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not read {path}: {exc}") from exc

    return _loaded_response(path.name, text)


@app.post("/api/program/apply")
def api_program_apply(body: ApplyBody) -> dict[str, Any]:
    try:
        return apply_program(body.receipt_id)
    except ProgramServiceError as exc:
        raise _program_http(exc) from exc


class SnapshotBody(BaseModel):
    """Optional client metadata for snapshot JSON (no sample dumps)."""

    receipt_id: str | None = None
    step: int | None = None
    active_block: int | None = None
    view: str | None = None
    png_filename: str | None = None


@app.get("/api/claim-hygiene")
def api_claim_hygiene(receipt_id: str | None = None) -> dict[str, Any]:
    """Standing prohibitions + live claim badges (always honest)."""
    receipt = None
    if receipt_id:
        try:
            receipt = load_receipt(receipt_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return claim_hygiene_payload(receipt)


@app.post("/api/snapshot")
def api_snapshot(body: SnapshotBody | None = None) -> dict[str, Any]:
    """Return JSON receipt-slice metadata for a snapshot export.

    Does not persist files or accept sim dumps. PNG capture is client-side.
    """
    body = body or SnapshotBody()
    receipt = None
    rid = body.receipt_id
    if rid:
        try:
            receipt = load_receipt(rid)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    slice_ = build_snapshot_slice(
        receipt,
        client={
            "receipt_id": rid,
            "step": body.step,
            "active_block": body.active_block,
            "view": body.view,
            "png_filename": body.png_filename,
        },
    )
    return {"ok": True, "version": APP_VERSION, "snapshot": slice_}


@app.get("/api/graph")
def get_graph() -> dict[str, Any]:
    return _engine.graph_payload()


@app.post("/api/reset")
def reset(body: ConfigBody) -> dict[str, Any]:
    cfg = SamplerConfig(**body.model_dump())
    return _engine.reset(cfg)


@app.post("/api/params")
def patch_params(body: ParamPatch) -> dict[str, Any]:
    return _engine.update_params(**body.model_dump(exclude_none=True))


@app.post("/api/sample")
def sample_once(n: int = 8) -> dict[str, Any]:
    return _engine.sample_batch(n_samples=n)


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    engine = SamplerEngine(SamplerConfig())
    running = False
    stop = False
    await ws.send_text(
        json.dumps(
            {
                "type": "status",
                "running": False,
                "connected": True,
                "message": "ws open",
            }
        )
    )

    async def sender() -> None:
        nonlocal running
        while not stop:
            if not running:
                await asyncio.sleep(0.05)
                continue
            try:
                # Run sampling in thread so event loop stays responsive
                batch = await asyncio.to_thread(engine.sample_batch)
                await ws.send_text(json.dumps(batch))
                await asyncio.sleep(0.02)
            except Exception as exc:  # noqa: BLE001
                await ws.send_text(json.dumps({"type": "error", "message": f"sampler: {exc}"}))
                running = False
                await asyncio.sleep(0.2)

    task = asyncio.create_task(sender())
    try:
        await ws.send_text(json.dumps({"type": "graph", **engine.graph_payload()}))
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "reset":
                cfg = _config_from_msg(msg)
                payload = engine.reset(cfg)
                running = False
                await ws.send_text(json.dumps({"type": "graph", **payload}))
            elif mtype == "params":
                payload = engine.update_params(
                    **{
                        k: msg[k]
                        for k in (
                            "beta",
                            "J",
                            "h",
                            "clamp",
                            "warmup",
                            "steps_per_sample",
                            "batch_size",
                            "seed",
                        )
                        if k in msg
                    }
                )
                await ws.send_text(json.dumps({"type": "graph", **payload}))
            elif mtype == "run":
                running = True
                await ws.send_text(json.dumps({"type": "status", "running": True}))
            elif mtype == "pause":
                running = False
                await ws.send_text(json.dumps({"type": "status", "running": False}))
            elif mtype == "step":
                batch = await asyncio.to_thread(
                    engine.sample_batch, n_samples=int(msg.get("n", 1))
                )
                await ws.send_text(json.dumps(batch))
            elif mtype == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        stop = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass



class AblationBody(BaseModel):
    """Zero a term and measure the change against a noise floor."""

    receipt_id: str
    target: str = Field("couplings", pattern="^(couplings|biases|none)$")
    n_samples: int = Field(128, ge=16, le=1024)
    seed: int = Field(0, ge=0, le=2**31 - 1)


@app.post("/api/ablation/run")
def api_ablation_run(body: AblationBody) -> dict[str, Any]:
    """Try to prove the loaded model's own terms are doing nothing.

    The comparison is against a NOISE FLOOR, the difference between two runs of
    the same model under different seeds, not against zero. See
    backend/app/ablation.py and audit/findings/R23.md.
    """
    from .ablation import run_ablation

    try:
        return {"ok": True, **run_ablation(
            body.receipt_id, target=body.target,
            n_samples=body.n_samples, seed=body.seed)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# The Alloy Distribution Lab was removed. It was a view hardwired to one
# 8x8 program, with three endpoints of its own, showing order parameters
# (staggered magnetization, short-range order, configuration energy) that
# are not alloy-specific at all: they are general readings any model can
# have. A bespoke lab per model is the opposite of a pluggable decoder.
#
# The MODEL stays. programs/alloy_ordering_8x8.yaml and its receipt are a
# known-answer test case: an antiferromagnetic checkerboard has a Bragg
# peak at (pi, pi), which is the cleanest possible check that a structure
# factor view works.

class EbmTrainBody(BaseModel):
    """Optional short retrain (lite) or no-op load of existing checkpoint."""

    lite: bool = True
    epochs: int = Field(12, ge=1, le=40)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    force: bool = False


class EbmSampleBody(BaseModel):
    """Sample from compiled RBM receipt (or numpy Gibbs fallback) and decode to 4×4 visibles."""

    n: int = Field(16, ge=1, le=64)
    beta: float | None = Field(None, ge=0.01, le=5.0)
    warmup: int = Field(40, ge=0, le=2000)
    steps_per_sample: int = Field(4, ge=1, le=32)
    seed: int | None = None
    receipt_id: str | None = EBM_RECEIPT_ID
    use_checkpoint_gibbs: bool = False


@app.post("/api/lab/ebm/train")
def api_ebm_train(body: EbmTrainBody) -> dict[str, Any]:
    """Load checkpoint by default; optional lite retrain (seconds, not minutes)."""
    ckpt = load_checkpoint()
    if ckpt is not None and not body.force:
        log = ckpt.get("train_log") or {}
        return {
            "ok": True,
            "action": "loaded_checkpoint",
            "final_cd_moment_l1": log.get("final_cd_moment_l1"),
            "pure_rate": log.get("pure_rate"),
            "mean_bar_stripe_score": log.get("mean_bar_stripe_score"),
            "n_epochs": log.get("n_epochs"),
            "curve": (log.get("curve") or [])[-40:],
            "artifact_dir": ckpt.get("artifact_dir"),
            "honesty": EBM_HONESTY,
            "model_kind": "rbm",
        }
    # Lite retrain for demo responsiveness (does not recompile receipt)
    result = train_and_export(n_epochs=body.epochs, lite=bool(body.lite), seed=body.seed)
    return {
        "ok": True,
        "action": "retrain_lite",
        "final_cd_moment_l1": result["final_cd_moment_l1"],
        "pure_rate": result.get("pure_rate"),
        "mean_bar_stripe_score": result.get("mean_bar_stripe_score"),
        "gate_passed": result.get("gate_passed"),
        "elapsed_s": result["elapsed_s"],
        "n_epochs": result["n_epochs"],
        "n_train": result.get("n_train"),
        "curve": result.get("train_log") or [],
        "paths": result.get("paths"),
        "honesty": EBM_HONESTY,
        "model_kind": "rbm",
        "note": "Lite retrain only, full train via scripts/train_bars_stripes_ebm.py --compile",
    }


@app.post("/api/lab/ebm/sample")
def api_ebm_sample(body: EbmSampleBody) -> dict[str, Any]:
    """Sample + decode bars-and-stripes images via SamplerEngine on compiled receipt."""
    rid = body.receipt_id or EBM_RECEIPT_ID
    ckpt = load_checkpoint()
    energies: list[float] | None = None
    states: list[list] = []

    if body.use_checkpoint_gibbs and ckpt is not None:
        from .ebm_bars_stripes import sample_model, spins_to_binary

        beta = float(body.beta) if body.beta is not None else 1.2
        spins = sample_model(
            ckpt["J_edge"],
            ckpt["h"],
            ckpt["edges"],
            n_samples=body.n,
            beta=beta,
            warmup=max(body.warmup, 80),
            seed=int(body.seed) if body.seed is not None else 0,
            W=ckpt.get("W"),
            a=ckpt.get("a"),
            b=ckpt.get("b"),
        )
        states = [spins_to_binary(s).astype(int).tolist() for s in spins]
    else:
        cfg = SamplerConfig(
            receipt_id=rid,
            size=EBM_GRID,
            beta=float(body.beta) if body.beta is not None else 1.0,
            warmup=body.warmup,
            steps_per_sample=body.steps_per_sample,
            batch_size=body.n,
            seed=int(body.seed) if body.seed is not None else 0,
        )
        try:
            _engine.reset(cfg)
            batch = _engine.sample_batch(n_samples=body.n, warmup=body.warmup)
            states = batch.get("states") or []
            energies = batch.get("energies")
        except Exception as exc:  # noqa: BLE001
            # Fallback to checkpoint Gibbs if receipt sample fails
            if ckpt is None:
                raise HTTPException(status_code=500, detail=f"sample failed: {exc}") from exc
            from .ebm_bars_stripes import sample_model, spins_to_binary

            beta = float(body.beta) if body.beta is not None else 1.2
            spins = sample_model(
                ckpt["J_edge"],
                ckpt["h"],
                ckpt["edges"],
                n_samples=body.n,
                beta=beta,
                warmup=max(body.warmup, 80),
                seed=int(body.seed) if body.seed is not None else 0,
                W=ckpt.get("W"),
                a=ckpt.get("a"),
                b=ckpt.get("b"),
            )
            states = [spins_to_binary(s).astype(int).tolist() for s in spins]

    if not states:
        raise HTTPException(status_code=400, detail="no states sampled")

    n_sites = EBM_GRID * EBM_GRID
    decoded = []
    j_edge = ckpt["J_edge"] if ckpt else None
    h = ckpt["h"] if ckpt else None
    edges = ckpt["edges"] if ckpt else None
    for i, s in enumerate(states):
        flat = list(s)[:n_sites]
        if len(flat) < n_sites:
            flat = flat + [0] * (n_sites - len(flat))
        e = float(energies[i]) if energies and i < len(energies) else None
        decoded.append(
            decode_ebm_sample(
                flat,
                width=EBM_GRID,
                height=EBM_GRID,
                ising_energy=e,
                sample_index=i,
                J_edge=j_edge,
                h=h,
                edges=edges,
            )
        )

    decoded_sorted = sorted(
        decoded, key=lambda d: float(d.get("bar_stripe_score") or 0.0), reverse=True
    )
    pure_frac = float(sum(1 for d in decoded if d.get("is_pure")) / max(1, len(decoded)))
    return {
        "ok": True,
        "receipt_id": rid,
        "n_sampled": len(decoded),
        "samples": decoded_sorted,
        "pure_fraction": pure_frac,
        "caption": EBM_CAPTION,
        "honesty": EBM_HONESTY,
    }


@app.get("/api/lab/ebm/info")
def api_ebm_info() -> dict[str, Any]:
    """Dataset, honesty badge, artifact paths for EBM Lab."""
    ckpt = load_checkpoint()
    log = (ckpt or {}).get("train_log") or {}
    grids = (ckpt or {}).get("sample_grids") or {}
    return {
        "ok": True,
        "name": "EBM Lab",
        "program": EBM_PROGRAM_NAME,
        "receipt_id": EBM_RECEIPT_ID,
        "caption": EBM_CAPTION,
        "honesty": EBM_HONESTY,
        "dataset": {
            "name": "bars_and_stripes",
            "grid": f"{EBM_GRID}x{EBM_GRID}",
            "n_visible": EBM_N_VISIBLE,
            "n_hidden": EBM_N_HIDDEN,
            "n_spins": EBM_N_VISIBLE + EBM_N_HIDDEN,
            "positive": "pure horizontal bars OR pure vertical stripes",
            "not": ["MNIST", "Extropic codon", "alloy", "pairwise-visible Ising"],
        },
        "model": {
            "kind": "Restricted Boltzmann Machine (RBM)",
            "graph": f"{EBM_GRID}x{EBM_GRID} visibles + {EBM_N_HIDDEN} hiddens; couplings only V↔H",
            "convention": "E(v,h)= -v^T W h - a^T v - b^T h ; s in {-1,+1}; Ising export on [v|h]",
            "expect": "bipartite V–H, deg(v)=n_h≤12, deg(h)=16, 0 mediators",
            "pairwise_failure": "Pairwise grid Ising cannot represent bars∪stripes (conflicting ferro) → blobs; CD L1 was false success.",
        },
        "train": {
            "final_cd_moment_l1": log.get("final_cd_moment_l1"),
            "pure_rate": log.get("pure_rate"),
            "mean_bar_stripe_score": log.get("mean_bar_stripe_score"),
            "gate_passed": log.get("gate_passed"),
            "n_epochs": log.get("n_epochs"),
            "curve": (log.get("curve") or [])[-40:],
            "checkpoint_loaded": ckpt is not None,
        },
        "paths": {
            "artifacts": str(ARTIFACT_DIR),
            "yaml": "programs/ebm_bars_stripes.yaml",
            "receipt": f"receipts/{EBM_RECEIPT_ID}/",
            "data_examples_png": "artifacts/ebm_bars_stripes/data_examples.png",
            "samples_png": "artifacts/ebm_bars_stripes/samples_after_train.png",
        },
        "cached_data_examples": grids.get("data") or [],
        "cached_model_samples": grids.get("samples") or [],
    }



def create_app() -> FastAPI:
    return app
