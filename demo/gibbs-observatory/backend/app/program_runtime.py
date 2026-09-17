"""Running a loaded program: build, compile once, then drive it frame by frame.

This is the spine of the vertical slice. A program declares `build(inputs)`,
`INPUT_PORTS` and `DECODER`; this turns that into something the viewport can
step.

The shape of the loop is set by one measured fact (R32): compiling the sampler
costs ~300 ms and sampling a frame costs ~0.45 ms, so the compile happens once
and every frame after it is nearly free. What makes that possible is that the
input arrives as DATA -- new biases on the same graph -- rather than as a new
program. A port declared `bias` can change every frame for free. A port
declared `clamp` can change its VALUES for free too, but not which spins are
clamped, because that is structure.

The session holds one `LiveSampler`, which holds one chain that carries forward
across frames. That is deliberate: a game resumes from the previous frame
rather than re-thermalising from noise, which is both cheaper and what keeps
the picture from boiling. It also means the decoded answer keeps improving for
a while after the input settles, which is a property of the chain rather than a
bug, and which the caller can see because `frames` is reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .program_contract import Program, ProgramError, load_program

#: Block-Gibbs sweeps per frame. Measured on the 1536-spin visibility model,
#: 4 and 8 sweeps both cost about 0.4 ms a frame and 32 costs 0.96 ms, so the
#: choice is not about frame budget at this size. 8 is a middle value that
#: responds quickly to changed input without spending time that buys nothing.
DEFAULT_SWEEPS = 8


class RuntimeError_(ProgramError):
    """Something went wrong driving a program, with the reason."""


@dataclass
class Frame:
    """One step of a running program."""

    spins: np.ndarray
    frames: int
    n_spins: int
    #: Whether the sampler has recompiled since it started. Always 1 for a
    #: healthy session; more means something changed structure behind our
    #: back, which costs ~300 ms a frame and is worth surfacing (R32).
    traces: int


def initial_inputs(program: Program, given: dict[str, Any] | None = None) -> dict[str, Any]:
    """Enough input to build a first frame, without the caller knowing the
    program's input format.

    Otherwise nothing can open a program it does not already understand, which
    would make the Workbench's own Open button impossible: it would have to
    know every program's ports before it could show them.

    Order of preference. Anything the caller supplied wins. Then the program's
    own `DEFAULT_INPUTS`, if it declares them, because only the author knows
    what a sensible opening state is. Then zeros of the declared shape, which
    is a real configuration rather than a guess -- for the visibility model it
    is an empty room -- and which is possible only because the port declares
    its shape.

    A port with no shape gets nothing, and the program's `build` may raise. The
    alternative is inventing a shape, and a model built on an invented shape
    would sample happily and mean nothing.
    """
    out: dict[str, Any] = {}
    declared = getattr(program.module, "DEFAULT_INPUTS", None)
    if isinstance(declared, dict):
        out.update(declared)
    for port in program.ports:
        if port.name in out:
            continue
        if port.shape:
            out[port.name] = np.zeros(port.shape, dtype=float)
    if given:
        out.update(given)
    return out


class Session:
    """One running program, with its sampler and its chain."""

    def __init__(self, program: Program, inputs: dict[str, Any], *,
                 sweeps: int = DEFAULT_SWEEPS, seed: int = 0) -> None:
        from tsu_compiler.backends.live import LiveSampler
        from tsu_compiler.passes.analyse import analyse
        from tsu_compiler.passes.program import build_program

        self.program = program
        # Ports the caller did not supply are filled from the program's
        # own defaults, or zeros of the declared shape.
        self.inputs = initial_inputs(program, inputs)

        try:
            model = program.build(self.inputs)
        except Exception as exc:  # noqa: BLE001 -- a program may raise anything
            raise RuntimeError_(
                f"{program.name}.build() raised: {type(exc).__name__}: "
                f"{exc}") from exc

        self.model = model
        self.report = analyse(model)
        if not self.report.bipartite and not getattr(self.report, "colour_blocks", 0):
            raise RuntimeError_(
                f"{program.name} produced a lattice the compiler cannot "
                f"colour, so block Gibbs has no schedule to run")

        try:
            self.sampler = LiveSampler(build_program(model, self.report),
                                       sweeps=sweeps, seed=seed)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError_(
                f"could not build a sampler for {program.name}: "
                f"{type(exc).__name__}: {exc}") from exc

        self.frames = 0

    # -- driving ----------------------------------------------------------

    def step(self, inputs: dict[str, Any] | None = None) -> Frame:
        """Advance one frame, optionally with new input.

        New input is folded in by rebuilding the model's PARAMETERS and handing
        the sampler the new biases. The graph is not rebuilt and the sampler is
        not recompiled; if either happened this would cost ~300 ms a frame
        instead of ~0.45 ms.
        """
        if inputs:
            self.inputs.update(inputs)
            try:
                model = self.program.build(self.inputs)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError_(
                    f"{self.program.name}.build() raised on new input: "
                    f"{type(exc).__name__}: {exc}") from exc

            # Structure must not have moved. If it did, the sampler we already
            # compiled describes a different model, and quietly sampling it
            # would produce confident numbers for the wrong graph.
            if len(model.nodes) != len(self.model.nodes) or \
                    model.edges != self.model.edges:
                raise RuntimeError_(
                    f"{self.program.name}.build() returned a different graph "
                    f"for this input ({len(self.model.nodes)} spins and "
                    f"{len(self.model.edges)} edges before, "
                    f"{len(model.nodes)} and {len(model.edges)} now). Input "
                    f"may change biases, not structure; a new graph needs a "
                    f"new session.")
            self.model = model
            self.sampler.step(biases=np.asarray(model.biases, dtype=float))
        else:
            self.sampler.step()

        self.frames += 1
        return Frame(
            spins=self.sampler.state,
            frames=self.frames,
            n_spins=self.sampler.n_spins,
            traces=self.sampler.traces,
        )

    def decode(self, frame: Frame, *, chain: int = 0) -> Any:
        """Hand the frame to the program's own decoder, if it has one.

        A program without a `decode` is not an error: its spins are still
        readable, they just have no pictorial form the program claims to know.
        """
        fn = getattr(self.program.module, "decode", None)
        if not callable(fn):
            return None
        shape = self._decoder_shape()
        try:
            return fn(frame.spins[chain], shape)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError_(
                f"{self.program.name}.decode() raised: "
                f"{type(exc).__name__}: {exc}") from exc

    def _decoder_shape(self) -> tuple[int, ...]:
        """The shape the decoder reads the spins in.

        This is the LATTICE shape, which a port shape is not. A port describes
        what goes in -- the visibility world takes a three-number pose -- and
        the decoder needs what comes out, which is a 48x32 lattice. Taking the
        first port's shape handed that decoder (3,) for 1536 spins.

        Preference: what the program declares, then a port whose shape happens
        to match the spin count (true of the program that takes occupancy
        directly, where input and lattice really are the same grid), then flat.
        """
        if self.program.lattice_shape:
            return tuple(self.program.lattice_shape)
        n = self.sampler.n_spins
        for port in self.program.ports:
            if port.shape and int(np.prod(port.shape)) == n:
                return tuple(port.shape)
        return (n,)


def open_session(path: str | Path, inputs: dict[str, Any], *,
                 consent: bool = False, sweeps: int = DEFAULT_SWEEPS,
                 seed: int = 0) -> Session:
    """Load a program and start it. `consent` is required; see program_contract.

    The compiler is put on the path FIRST, before the program is executed. A
    program builds an IsingModel, so it imports tsu_compiler, and in a running
    server nothing has necessarily imported it yet -- `ensure_tsu_importable`
    is what finds it in a repo checkout, a pip install, or a frozen bundle.
    Without this the program raises ModuleNotFoundError at exec time and the
    reader is told their file is broken when the application simply had not
    set the path up.

    This was invisible to the tests, which insert `src` into sys.path in their
    own module headers and so were more permissive than the real thing.
    """
    from .program_service import ensure_tsu_importable
    ensure_tsu_importable()
    program = load_program(path, consent=consent)
    return Session(program, inputs, sweeps=sweeps, seed=seed)
