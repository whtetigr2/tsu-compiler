"""A loadable program: code that BUILDS a model, not a file that is one.

Slice 2's door reads a .py as data -- a literal SPEC dict, lifted out with
`ast.literal_eval`, never executed. That covers a model someone wrote down.

It cannot cover the game. The game's biases are rewritten every frame from
where the player is standing, so no file IS its model; something has to build
one on demand. That needs real code, and real code runs.

These two things share a file extension and nothing else, so the door has to
tell them apart and say which it found. Handing a program file the data door's
refusal -- "SPEC is not a literal" -- describes a problem the author does not
have.

ON CLAMPING, which the design document gets wrong. Section 9 says input is
clamping: pin some spins, let the rest relax. That is true of the interface in
general and FALSE of the one program that is actually verified. The visibility
model takes its input as BIASES and clamps nothing, deliberately, because the
clamped version made the couplings inert and R23 is the retraction. A contract
that called every input a clamp would encode the retracted design. So a port
declares how it enters, and `bias` is a first-class answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from backend.app.program_contract import (  # noqa: E402
    ConsentRequired,
    InputPort,
    ProgramError,
    describe_source,
    load_program,
)

GOOD = '''
"""A tiny program, in the shape the Workbench asks for."""
import numpy as np
from tsu_compiler.preflight.model import IsingModel

NAME = "two_cell_demo"
DECODER = "grid"
INPUT_PORTS = [
    {"name": "occupancy", "shape": [2, 2], "mode": "bias",
     "doc": "1 where a cell is solid"},
]


def build(inputs):
    occ = np.asarray(inputs["occupancy"], dtype=float).reshape(-1)
    n = occ.size
    edges = tuple((i, i + 1) for i in range(n - 1))
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(n)),
        edges=edges,
        weights=np.full(len(edges), 0.4),
        biases=-2.0 * occ,
        beta=1.0,
        offset=0.0,
    )
'''


def _write(tmp_path: Path, text: str, name: str = "prog.py") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestConsent:
    """Loading a program runs a stranger's code. That is the deal, and it is
    the same deal Blender makes, but it is never made silently."""

    def test_loading_without_consent_is_refused(self, tmp_path):
        p = _write(tmp_path, GOOD)
        with pytest.raises(ConsentRequired):
            load_program(p)

    def test_the_refusal_names_the_file_and_says_it_executes(self, tmp_path):
        p = _write(tmp_path, GOOD)
        with pytest.raises(ConsentRequired) as e:
            load_program(p)
        msg = str(e.value)
        assert "prog.py" in msg
        assert "run" in msg.lower() or "execut" in msg.lower()

    def test_it_does_not_claim_a_sandbox(self, tmp_path):
        p = _write(tmp_path, GOOD)
        with pytest.raises(ConsentRequired) as e:
            load_program(p)
        msg = str(e.value).lower()
        assert "sandbox" not in msg and "safely" not in msg

    def test_consent_actually_loads_it(self, tmp_path):
        prog = load_program(_write(tmp_path, GOOD), consent=True)
        assert prog.name == "two_cell_demo"

    def test_nothing_runs_before_consent_is_given(self, tmp_path):
        """The refusal must come from inspecting the file, not from running it
        and then apologising."""
        marker = tmp_path / "ran.txt"
        src = GOOD + f"\nopen(r'{marker}', 'w').write('ran')\n"
        with pytest.raises(ConsentRequired):
            load_program(_write(tmp_path, src))
        assert not marker.exists(), "the module executed before consent"


class TestTheContract:
    def test_it_exposes_build_ports_and_decoder(self, tmp_path):
        prog = load_program(_write(tmp_path, GOOD), consent=True)
        assert callable(prog.build)
        assert prog.decoder == "grid"
        assert len(prog.ports) == 1

    def test_ports_are_parsed_into_records(self, tmp_path):
        prog = load_program(_write(tmp_path, GOOD), consent=True)
        port = prog.ports[0]
        assert isinstance(port, InputPort)
        assert port.name == "occupancy"
        assert port.shape == (2, 2)
        assert port.mode == "bias"
        assert port.doc

    def test_build_returns_a_model_the_compiler_understands(self, tmp_path):
        prog = load_program(_write(tmp_path, GOOD), consent=True)
        model = prog.build({"occupancy": [[1, 0], [0, 1]]})
        assert len(model.nodes) == 4
        assert len(model.edges) == 3

    def test_a_file_without_build_is_refused_by_name(self, tmp_path):
        src = GOOD.replace("def build(inputs):", "def make(inputs):")
        with pytest.raises(ProgramError) as e:
            load_program(_write(tmp_path, src), consent=True)
        assert "build" in str(e.value)

    def test_build_that_is_not_callable_is_refused(self, tmp_path):
        with pytest.raises(ProgramError) as e:
            load_program(_write(tmp_path, "build = 3\nINPUT_PORTS = []\n"),
                         consent=True)
        assert "callable" in str(e.value).lower()

    def test_a_program_with_no_ports_is_allowed(self, tmp_path):
        """A model with no external input is a perfectly good program. It just
        cannot be driven."""
        src = GOOD.replace("INPUT_PORTS = [\n    {\"name\": \"occupancy\", "
                           "\"shape\": [2, 2], \"mode\": \"bias\",\n     "
                           "\"doc\": \"1 where a cell is solid\"},\n]",
                           "INPUT_PORTS = []")
        prog = load_program(_write(tmp_path, src), consent=True)
        assert prog.ports == ()


class TestPortModes:
    """The part the design document has backwards."""

    def test_bias_is_a_valid_mode(self, tmp_path):
        prog = load_program(_write(tmp_path, GOOD), consent=True)
        assert prog.ports[0].mode == "bias"

    def test_clamp_is_a_valid_mode(self, tmp_path):
        src = GOOD.replace('"mode": "bias"', '"mode": "clamp"')
        prog = load_program(_write(tmp_path, src), consent=True)
        assert prog.ports[0].mode == "clamp"

    def test_an_unknown_mode_is_refused_with_the_options(self, tmp_path):
        src = GOOD.replace('"mode": "bias"', '"mode": "wiggle"')
        with pytest.raises(ProgramError) as e:
            load_program(_write(tmp_path, src), consent=True)
        msg = str(e.value)
        assert "wiggle" in msg and "bias" in msg and "clamp" in msg

    def test_a_port_without_a_mode_is_refused_rather_than_defaulted(self, tmp_path):
        """Defaulting would pick a side of R23 silently. Clamping input made
        the couplings inert once already; the author says which they mean."""
        src = GOOD.replace('"mode": "bias",\n     ', "")
        with pytest.raises(ProgramError) as e:
            load_program(_write(tmp_path, src), consent=True)
        assert "mode" in str(e.value)


class TestFailuresAreReadable:
    def test_a_syntax_error_names_the_line(self, tmp_path):
        with pytest.raises(ProgramError) as e:
            load_program(_write(tmp_path, "def build(\n"), consent=True)
        assert "line" in str(e.value).lower()

    def test_an_exception_at_import_is_attributed_to_the_program(self, tmp_path):
        src = "raise RuntimeError('boom')\n"
        with pytest.raises(ProgramError) as e:
            load_program(_write(tmp_path, src), consent=True)
        msg = str(e.value)
        assert "boom" in msg
        assert "prog.py" in msg

    def test_a_missing_file_is_refused_before_consent_matters(self, tmp_path):
        with pytest.raises(ProgramError):
            load_program(tmp_path / "nope.py", consent=True)


class TestTellingTheTwoKindsApart:
    """The door must route, not guess. A data .py and a program .py share an
    extension and nothing else."""

    def test_a_program_file_is_recognised_as_a_program(self, tmp_path):
        assert describe_source(GOOD).kind == "program"

    def test_a_literal_spec_file_is_recognised_as_data(self, tmp_path):
        assert describe_source("SPEC = {'name': 'x'}\n").kind == "data"

    def test_a_file_that_is_neither_says_so(self, tmp_path):
        assert describe_source("x = 1\n").kind == "neither"

    def test_recognising_a_program_does_not_execute_it(self, tmp_path):
        marker = tmp_path / "ran2.txt"
        describe_source(GOOD + f"\nopen(r'{marker}', 'w').write('ran')\n")
        assert not marker.exists()

    def test_a_file_with_both_is_called_a_program(self, tmp_path):
        """build() can do things a literal cannot, so it is the stronger
        claim about what the author meant."""
        src = GOOD + "\nSPEC = {'name': 'also_here'}\n"
        assert describe_source(src).kind == "program"

    def test_the_description_says_why(self, tmp_path):
        d = describe_source(GOOD)
        assert "build" in d.why


class TestBuildMustTakeInputs:
    """`build(inputs)` is the contract, so the entry point takes inputs.

    A zero-argument helper that happens to be named `build` is not a program,
    and the distinction is load-bearing rather than pedantic: the data door's
    own case is `def build(): return {}` with `SPEC = build()`, which is a
    computed literal with a helper. Calling that a program would route it away
    from the refusal that actually describes it.
    """

    def test_a_zero_argument_build_is_not_a_program(self):
        src = "def build():\n    return {}\nSPEC = build()\n"
        assert describe_source(src).kind == "data"

    def test_a_one_argument_build_is_a_program(self):
        assert describe_source("def build(inputs):\n    return None\n").kind == "program"

    def test_star_args_counts(self):
        assert describe_source("def build(*a):\n    return None\n").kind == "program"

    def test_a_keyword_only_argument_counts(self):
        src = "def build(*, inputs):\n    return None\n"
        assert describe_source(src).kind == "program"

    def test_a_zero_argument_build_with_no_spec_is_neither(self):
        assert describe_source("def build():\n    return {}\n").kind == "neither"
