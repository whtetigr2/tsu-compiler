"""A loadable program: code that BUILDS a model, not a file that is one.

`spec_formats` reads a .py as data -- a literal `SPEC` dict, lifted out with
`ast.literal_eval` and never executed. That covers a model someone wrote down,
and it is the right default, because opening a file is not permission to run it.

It cannot cover the game. The game's biases are rewritten every frame from
where the player is standing, so no single file IS its model; something has to
build one on demand. That needs real code, and real code runs. Blender made the
same decision for the same reason.

A program declares three things:

    build(inputs) -> IsingModel     makes the lattice
    INPUT_PORTS                     what external input it accepts
    DECODER                         how what pops out becomes a picture

## On consent

Loading one runs a stranger's code, with this process's privileges, and there
is no clever way around that. So: never without explicit consent, the prompt
names the file and says plainly that it executes, and nothing is inspected by
running it -- `describe_source` reads the AST, so the Workbench can tell the
reader what a file is before asking whether to run it. No sandbox is claimed,
because none exists.

## On clamping, where the design document is wrong

Section 9 of the spec says input is clamping: pin some spins, let the rest
relax. That is true of the interface in general and false of the one program
here that is actually verified against ground truth. `audit/visibility_on_thrml.py`
takes its input as BIASES and clamps nothing, deliberately:

    "A wall is a bias of -W_WALL, free space +W_FREE. Nothing is clamped, so
     the couplings are able to outvote the input -- which is the entire point,
     and was impossible in the clamped version."

Clamping the input is what made the couplings inert, and R23 is the
retraction. A contract that called every input a clamp would encode the
retracted design and quietly re-introduce the defect. So a port declares its
mode, `bias` and `clamp` are both first-class, and there is no default --
defaulting would pick a side of R23 without the author noticing.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

#: How external input reaches the model. Neither is the default; see above.
PORT_MODES = ("bias", "clamp")

#: Module-level names a program may use for each declaration, in search order.
BUILD_NAMES = ("build",)
PORTS_NAMES = ("INPUT_PORTS", "input_ports")
DECODER_NAMES = ("DECODER", "decoder")
NAME_NAMES = ("NAME", "name")

#: What `spec_formats` looks for when reading a .py as data.
DATA_NAMES = ("SPEC", "spec", "WORKLOAD", "workload", "PROGRAM", "program")


class ProgramError(ValueError):
    """The file is not a program this Workbench can load, with the reason."""


class ConsentRequired(ProgramError):
    """Loading would execute the file and nobody has said yes."""


@dataclass(frozen=True)
class InputPort:
    """One channel of external input, and how it enters the model."""

    name: str
    shape: tuple[int, ...]
    #: "bias" or "clamp". Declared, never inferred.
    mode: str
    doc: str = ""


@dataclass(frozen=True)
class Program:
    name: str
    build: Callable[[dict[str, Any]], Any]
    ports: tuple[InputPort, ...]
    decoder: str
    source: Path
    module: ModuleType


@dataclass(frozen=True)
class SourceKind:
    """What a .py appears to be, decided without running it."""

    #: "program", "data", or "neither".
    kind: str
    why: str


def _takes_an_argument(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether this def could be called as build(inputs)."""
    a = node.args
    return bool(a.posonlyargs or a.args or a.vararg or a.kwonlyargs)


def describe_source(text: str) -> SourceKind:
    """Decide what a .py is by reading it, never by running it.

    This is what lets the Workbench say "this file contains code and will
    execute" BEFORE asking. Deciding by import would mean running the file to
    find out whether running it was acceptable.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return SourceKind("neither", f"it does not parse: {e.msg} at line {e.lineno}")

    has_build = False
    assigned: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The contract is build(inputs), so the entry point TAKES inputs.
            # A zero-argument helper that happens to be called `build` is not
            # a program entry point, and the distinction is load-bearing: the
            # data door's own test case is `def build(): return {}` followed
            # by `SPEC = build()`, which is a computed literal with a helper,
            # not code that makes a model on demand.
            if node.name in BUILD_NAMES and _takes_an_argument(node):
                has_build = True
            continue
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name):
                assigned.add(t.id)
                if t.id in BUILD_NAMES:
                    has_build = True

    if has_build:
        # `build()` can do what a literal cannot, so its presence is the
        # stronger claim about what the author meant, even alongside a SPEC.
        return SourceKind("program", "it defines build(), so it is code that "
                                     "makes a model rather than a model")
    if assigned & set(DATA_NAMES):
        found = ", ".join(sorted(assigned & set(DATA_NAMES)))
        return SourceKind("data", f"it assigns {found} and defines no build(), "
                                  f"so it can be read without running it")
    return SourceKind("neither", "it defines no build() and assigns none of "
                                 + ", ".join(DATA_NAMES))


def _first(module: ModuleType, names: tuple[str, ...]) -> Any:
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None


def _parse_port(raw: Any, index: int) -> InputPort:
    if isinstance(raw, InputPort):
        port = raw
        if port.mode not in PORT_MODES:
            raise ProgramError(
                f"input port {port.name!r} declares mode {port.mode!r}; "
                f"the modes are {' and '.join(PORT_MODES)}")
        return port

    if not isinstance(raw, dict):
        raise ProgramError(
            f"INPUT_PORTS[{index}] is a {type(raw).__name__}; each port is a "
            f"mapping with name, shape and mode")

    name = raw.get("name")
    if not name:
        raise ProgramError(f"INPUT_PORTS[{index}] has no name")

    if "mode" not in raw:
        # Deliberately not defaulted. Clamping the input is what made the
        # couplings inert in R23, and a default would pick a side of that
        # without the author noticing.
        raise ProgramError(
            f"input port {name!r} does not declare a mode. Say "
            f"{' or '.join(repr(m) for m in PORT_MODES)}: a bias lets the "
            f"model's couplings outvote the input, a clamp does not. There is "
            f"no default, because clamping input is what made the couplings "
            f"inert in R23.")

    mode = raw["mode"]
    if mode not in PORT_MODES:
        raise ProgramError(
            f"input port {name!r} declares mode {mode!r}; the modes are "
            f"{' and '.join(repr(m) for m in PORT_MODES)}")

    shape = raw.get("shape", ())
    try:
        shape = tuple(int(x) for x in shape)
    except (TypeError, ValueError) as e:
        raise ProgramError(
            f"input port {name!r} has shape {shape!r}, which is not a "
            f"sequence of integers") from e

    return InputPort(name=str(name), shape=shape, mode=str(mode),
                     doc=str(raw.get("doc", "")))


def load_program(path: str | Path, *, consent: bool = False) -> Program:
    """Execute a program file and return what it declares.

    `consent` must be True. It is a parameter rather than a prompt so that the
    decision is made by whoever is talking to the user, and so it appears in
    every call site as a visible fact rather than an assumption.
    """
    path = Path(path)
    if not path.is_file():
        raise ProgramError(f"no program file at {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProgramError(f"could not read {path.name}: {exc}") from exc

    if not consent:
        # Checked before anything executes, and described from the AST, so the
        # reader learns what the file is without it having run.
        described = describe_source(text)
        raise ConsentRequired(
            f"{path.name} contains code and loading it will run that code on "
            f"this machine, with the same access this application has. "
            f"({described.why}.) Nothing has been run. Load it only if you "
            f"trust where it came from.")

    module = ModuleType(f"workbench_program_{path.stem}")
    module.__file__ = str(path)
    # The program's own directory goes on the path so it can import helpers
    # next to it, the way the audit scripts already do.
    added = str(path.parent)
    inserted = added not in sys.path
    if inserted:
        sys.path.insert(0, added)
    try:
        code = compile(text, str(path), "exec")
    except SyntaxError as exc:
        raise ProgramError(
            f"{path.name} does not parse: {exc.msg} at line {exc.lineno}") from exc

    try:
        exec(code, module.__dict__)  # noqa: S102 -- the whole point, with consent
    except Exception as exc:  # noqa: BLE001 -- a program may raise anything
        raise ProgramError(
            f"{path.name} raised while loading: "
            f"{type(exc).__name__}: {exc}") from exc
    finally:
        if inserted:
            try:
                sys.path.remove(added)
            except ValueError:
                pass

    build = _first(module, BUILD_NAMES)
    if build is None:
        raise ProgramError(
            f"{path.name} defines no build(). A program declares "
            f"build(inputs) -> IsingModel, INPUT_PORTS, and DECODER.")
    if not callable(build):
        raise ProgramError(
            f"build in {path.name} is a {type(build).__name__} and must be "
            f"callable: build(inputs) -> IsingModel")

    raw_ports = _first(module, PORTS_NAMES)
    if raw_ports is None:
        raw_ports = []
    if not isinstance(raw_ports, (list, tuple)):
        raise ProgramError(
            f"INPUT_PORTS in {path.name} is a {type(raw_ports).__name__} and "
            f"must be a list of ports")
    ports = tuple(_parse_port(p, i) for i, p in enumerate(raw_ports))

    seen: set[str] = set()
    for p in ports:
        if p.name in seen:
            raise ProgramError(f"{path.name} declares two input ports named "
                               f"{p.name!r}")
        seen.add(p.name)

    decoder = _first(module, DECODER_NAMES)
    name = _first(module, NAME_NAMES) or path.stem

    return Program(
        name=str(name),
        build=build,
        ports=ports,
        decoder=str(decoder) if decoder else "",
        source=path,
        module=module,
    )
