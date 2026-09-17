"""Read a thermodynamic program someone already wrote, in whatever they wrote it in.

The Workbench asks people to point it at a file. They will point it at YAML,
at JSON, at a .py that builds a spec, and at a .txt they pasted something into.
All four should work where they can, and where they cannot the reader should
learn why in one sentence.

Three rules hold everywhere below.

**Never guess silently.** Every load reports the format it decided on and what
decided it. A file whose name says one thing and whose content says another is
read by its content, and the report says so.

**Never execute a stranger's file.** A .py is parsed as data with `ast` and a
literal assignment is lifted out of it. `SPEC = build()` is refused, because
producing that value means running the module, and "load this file" is not
consent to run it. This is the only reason the Python path is restricted, and
it is not negotiable for convenience.

**Parsing is not being a program.** A file can be flawless YAML and still not
be a thermodynamic program. Those are different failures with different fixes,
so they get different messages.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import yaml

from .program_contract import describe_source

#: Module-level names a Python file may bind its spec to, in search order.
SPEC_NAMES = ("SPEC", "spec", "WORKLOAD", "workload", "PROGRAM", "program")

#: A parsed mapping needs one of these to be a workload the compiler can read.
#: Mirrors `tsu_compiler.spec.load_spec`, which branches on `generate` and
#: otherwise requires `variables`.
PROGRAM_KEYS = ("generate", "variables")


class DetectionError(ValueError):
    """The file could not be turned into a program, with the reason attached."""


@dataclass(frozen=True)
class Detection:
    """What the loader decided this file is, and what decided it."""

    format: str
    why: str
    #: Always False. Present so the answer to "did you run my file?" is a field
    #: rather than a promise in a docstring.
    executed: bool = False


@dataclass(frozen=True)
class Loaded:
    detection: Detection
    spec: dict[str, Any]
    #: YAML text for the editor. The compiler's loader reads YAML, so this is
    #: exactly what will be compiled, visible and editable before it is.
    yaml: str
    message: str


def detect(filename: str, text: str) -> Detection:
    """Decide what this file is. Content outranks the extension."""
    suffix = PurePosixPath(filename or "").suffix.lower()
    stripped = text.strip()

    if stripped.startswith(("{", "[")):
        if suffix == ".json":
            return Detection("json", "the .json extension and the content agree")
        return Detection(
            "json", f"the content starts with {stripped[0]!r}, so it is read as "
            f"JSON regardless of the {suffix or 'missing'} extension")

    if suffix == ".py":
        return Detection("python", "the .py extension; read as data, never run")
    if suffix == ".json":
        return Detection(
            "json", "the .json extension, though the content does not start "
            "with a brace")
    if suffix in (".yaml", ".yml"):
        return Detection("yaml", f"the {suffix} extension")
    return Detection(
        "yaml", f"no decisive extension ({suffix or 'none'}) and content that "
        f"is not JSON, so it is read as YAML")


def _describe_non_literal(node: ast.expr) -> str:
    """Say in plain words why a value is not a literal.

    `ast.literal_eval` raises with the offending node's full AST repr attached,
    so the reader gets `DictComp(key=JoinedStr(values=[Constant(value='v',
    kind=None), FormattedValue(...` where a short phrase would do. Find the
    first construct that stops the value being a literal and name it.
    """
    named = (
        (ast.DictComp, "built by a dict comprehension"),
        (ast.ListComp, "built by a list comprehension"),
        (ast.SetComp, "built by a set comprehension"),
        (ast.GeneratorExp, "built by a generator expression"),
        (ast.Call, "the result of a function call"),
        (ast.JoinedStr, "built with an f-string"),
        (ast.Attribute, "read off another object"),
        (ast.Subscript, "indexed out of another value"),
        (ast.IfExp, "chosen by a conditional expression"),
        (ast.BinOp, "computed by an expression"),
        (ast.Lambda, "a lambda"),
    )
    for sub in ast.walk(node):
        for kind, phrase in named:
            if isinstance(sub, kind):
                return phrase
        if isinstance(sub, ast.Name):
            return f"a reference to {sub.id}"
    return "not a literal"


def _python_spec(text: str) -> dict[str, Any]:
    """Lift a literal spec assignment out of Python source without running it.

    `ast.parse` builds a tree; `ast.literal_eval` evaluates only literals. A
    call, a name, or a comprehension raises, which is the refusal we want.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise DetectionError(
            f"this .py file does not parse: {e.msg} at line {e.lineno}") from e

    # A .py that defines build() is a PROGRAM: code that makes a model on
    # demand, which is the only way to express something like the game, whose
    # biases are rewritten every frame. It is not malformed data, and handing
    # its author the data door's complaint -- "SPEC is not a literal" --
    # describes a problem they do not have and points at a fix that would
    # destroy what they wrote. Route it, and say what it is.
    kind = describe_source(text)
    if kind.kind == "program":
        raise DetectionError(
            f"this file defines build(), so it is a program rather than a "
            f"model: code that builds a model on demand. Opening it as data "
            f"would lose that. Loading a program runs the file on this "
            f"machine, so the Workbench asks first.")

    found: list[tuple[str, ast.expr]] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id in SPEC_NAMES:
                assert node.value is not None
                found.append((t.id, node.value))

    if not found:
        raise DetectionError(
            f"this .py file defines no spec to read. The Workbench looks for a "
            f"module-level assignment named one of {', '.join(SPEC_NAMES)} "
            f"whose value is a literal, for example "
            f"SPEC = {{'generate': {{'grid': [4, 4]}}, 'terms': []}}. "
            f"It does not import or run the file, so a spec that is built by "
            f"code cannot be read this way.")

    name, value = found[-1]
    try:
        spec = ast.literal_eval(value)
    except ValueError as e:
        raise DetectionError(
            f"{name} in this .py file is {_describe_non_literal(value)}, so "
            f"reading it would mean running the file, and loading a file is "
            f"not permission to execute it. Replace it with a literal dict, "
            f"or save the spec as YAML or JSON.") from e
    if not isinstance(spec, dict):
        raise DetectionError(
            f"{name} in this .py file is a {type(spec).__name__}, and a "
            f"program must be a mapping")
    return spec


def _parse(fmt: str, text: str) -> dict[str, Any]:
    if not text.strip():
        raise DetectionError("this file is empty, so there is nothing to compile")

    if fmt == "python":
        return _python_spec(text)

    if fmt == "json":
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as e:
            raise DetectionError(
                f"this file is not valid JSON: {e.msg} at line {e.lineno}, "
                f"column {e.colno}") from e
    else:
        try:
            spec = yaml.safe_load(text)
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            problem = getattr(e, "problem", None) or str(e)
            raise DetectionError(f"this file is not valid YAML: {problem}{where}") from e

    if spec is None:
        raise DetectionError("this file parsed as empty, so there is nothing to compile")
    if not isinstance(spec, dict):
        raise DetectionError(
            f"a program is a mapping with a {' or '.join(PROGRAM_KEYS)} block, "
            f"and this file's top level is a {type(spec).__name__}")
    return spec


def _check_shape(spec: dict[str, Any]) -> None:
    """Parsed cleanly is not the same as being a program, and the difference is
    worth a distinct message: one is a typo, the other is the wrong file."""
    if not any(k in spec for k in PROGRAM_KEYS):
        keys = ", ".join(sorted(spec)[:6]) or "nothing"
        raise DetectionError(
            f"this file parses, but it is not a thermodynamic program: it has "
            f"no `variables` block and no `generate` block. Its top-level keys "
            f"are {keys}.")

    # `tsu_compiler.spec._generated_terms` reads gen["kind"] on its first line,
    # so a block without one raises a bare KeyError from inside the compiler
    # after the reader has already pressed Compile and waited. The requirement
    # is generic across every generator, so checking it here costs nothing and
    # turns a stack trace into a sentence. Which kinds exist is the compiler's
    # business, not this door's, so no list is asserted.
    gen = spec.get("generate")
    if gen is not None:
        if not isinstance(gen, dict):
            raise DetectionError(
                f"`generate` must be a mapping describing one shape, and here "
                f"it is a {type(gen).__name__}. It needs at least a `kind`, "
                f"for example:\n"
                f"  generate:\n    kind: grid\n    width: 8\n    height: 8\n"
                f"    variable_domain: binary")
        if "kind" not in gen:
            keys = ", ".join(sorted(gen)) or "nothing"
            raise DetectionError(
                f"the `generate` block has no `kind`, so the compiler cannot "
                f"tell which shape to build. It holds {keys}. A grid looks "
                f"like:\n"
                f"  generate:\n    kind: grid\n    width: 8\n    height: 8\n"
                f"    variable_domain: binary")


def _check_against_compiler(out_yaml: str) -> None:
    """Ask the compiler's own loader whether this is a spec it can read.

    The alternative is restating the compiler's rules here, which would drift
    the moment either side changed and would be wrong in exactly the cases
    that matter. Delegating costs one parse and is exact.

    The failure this catches is worth naming. `variable_domain: binary` is the
    obvious thing to write and the compiler wants `variable_domain:
    {domain: binary}`, so the obvious version raises `TypeError: string
    indices must be integers` from deep inside term construction, after the
    reader has pressed Compile and waited through placement. Here it lands at
    the moment they open the file.

    The compiler may not be importable at all -- `tsu_status()` tracks exactly
    that -- and a missing compiler is not a reason to refuse a file. Skip the
    check rather than fail it.
    """
    try:
        from tsu_compiler.spec import load_spec_text
    except Exception:
        return

    try:
        load_spec_text(out_yaml)
    except Exception as exc:
        raise DetectionError(
            f"this parses as a program but the compiler cannot read it: "
            f"{type(exc).__name__}: {exc}") from exc


def load_text(filename: str, text: str) -> Loaded:
    """The whole path: decide the format, parse it, check it is a program, and
    hand back YAML the reader can see before compiling it."""
    detection = detect(filename, text)
    spec = _parse(detection.format, text)
    _check_shape(spec)

    if detection.format == "yaml":
        # Round-tripping YAML through a parser destroys comments and key order,
        # and the author wrote those on purpose. Pass the original through.
        out_yaml = text
    else:
        out_yaml = yaml.safe_dump(spec, sort_keys=False, default_flow_style=False)

    _check_against_compiler(out_yaml)

    name = filename or "untitled"
    if detection.format == "yaml":
        message = f"loaded {name}"
    elif detection.format == "python":
        message = (f"loaded {name}, read as Python data and converted to YAML. "
                   f"The file was not run.")
    else:
        message = f"loaded {name}, converted from {detection.format.upper()} to YAML"

    return Loaded(detection=detection, spec=spec, yaml=out_yaml, message=message)
