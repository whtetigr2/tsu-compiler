"""Loading a program someone already wrote, in whatever they wrote it in.

The rule this file exists to enforce: the loader never guesses silently. Every
result says which format it decided on and why, and a file it cannot read is a
refusal with a reason, not an empty editor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The compiler-delegation check below is only a control if it can actually
# fail, so put the compiler on the path the same way test_extropic_oracle.py
# does rather than letting the test quietly skip.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from app.spec_formats import DetectionError, detect, load_text  # noqa: E402

GRID_YAML = """\
name: demo
generate:
  kind: grid
  width: 4
  height: 4
  variable_domain: {domain: binary}
terms: []
"""


class TestFormatDetection:
    def test_yaml_by_extension(self):
        d = detect("model.yaml", GRID_YAML)
        assert d.format == "yaml"
        assert d.executed is False

    def test_yml_is_yaml(self):
        assert detect("model.yml", GRID_YAML).format == "yaml"

    def test_json_by_extension(self):
        d = detect("model.json", '{"name": "demo", "generate": {"kind": "grid", "width": 4, "height": 4, "variable_domain": {"domain": "binary"}}, "terms": []}')
        assert d.format == "json"

    def test_json_by_content_when_extension_lies(self):
        """A .txt holding JSON is JSON. The content decides, and the
        detection says so rather than trusting the name."""
        d = detect("notes.txt", '{"name": "demo", "generate": {"kind": "grid", "width": 2, "height": 2, "variable_domain": {"domain": "binary"}}, "terms": []}')
        assert d.format == "json"
        assert "content" in d.why.lower()

    def test_bare_text_is_yaml(self):
        d = detect("scratch", GRID_YAML)
        assert d.format == "yaml"

    def test_python_is_detected_but_never_executed(self):
        src = "SPEC = {'name': 'demo', 'generate': {'kind': 'grid', 'width': 2, 'height': 2, 'variable_domain': {'domain': 'binary'}}, 'terms': []}\n"
        d = detect("model.py", src)
        assert d.format == "python"
        assert d.executed is False


class TestPythonExtraction:
    """A .py file is read as data, never run. That is the whole contract."""

    def test_literal_spec_dict_is_extracted(self):
        src = "SPEC = {'name': 'demo', 'generate': {'kind': 'grid', 'width': 3, 'height': 3, 'variable_domain': {'domain': 'binary'}}, 'terms': []}\n"
        out = load_text("model.py", src)
        assert out.spec["generate"]["width"] == 3
        assert out.detection.executed is False

    def test_lowercase_spec_name_also_works(self):
        src = "spec = {'name': 'demo', 'generate': {'kind': 'grid', 'width': 2, 'height': 2, 'variable_domain': {'domain': 'binary'}}, 'terms': []}\n"
        assert load_text("model.py", src).spec["generate"]["width"] == 2

    def test_imports_and_other_code_are_ignored_not_run(self):
        src = (
            "import os\n"
            "def helper():\n"
            "    return 1\n"
            "SPEC = {'name': 'demo', 'generate': {'kind': 'grid', 'width': 2, 'height': 2, 'variable_domain': {'domain': 'binary'}}, 'terms': []}\n"
        )
        assert load_text("model.py", src).spec["generate"]["width"] == 2

    def test_computed_spec_is_refused_with_a_reason(self):
        """`SPEC = build()` would need the file to run. Refuse and say so
        rather than importing a stranger's module."""
        src = "def build():\n    return {}\nSPEC = build()\n"
        with pytest.raises(DetectionError) as e:
            load_text("model.py", src)
        msg = str(e.value).lower()
        assert "literal" in msg
        assert "run" in msg or "execute" in msg

    def test_no_spec_assignment_is_refused_by_name(self):
        src = "x = 1\ny = 2\n"
        with pytest.raises(DetectionError) as e:
            load_text("model.py", src)
        assert "SPEC" in str(e.value)

    def test_syntax_error_reports_the_line(self):
        with pytest.raises(DetectionError) as e:
            load_text("model.py", "SPEC = {'a':\n")
        assert "line" in str(e.value).lower()


class TestParsing:
    def test_json_round_trips_to_a_spec(self):
        out = load_text("m.json", '{"name": "demo", "generate": {"kind": "grid", "width": 4, "height": 4, "variable_domain": {"domain": "binary"}}, "terms": []}')
        assert out.spec["generate"]["width"] == 4

    def test_json_error_names_the_position(self):
        with pytest.raises(DetectionError) as e:
            load_text("m.json", '{"generate": }')
        assert "line" in str(e.value).lower() or "column" in str(e.value).lower()

    def test_yaml_error_is_reported_not_swallowed(self):
        with pytest.raises(DetectionError) as e:
            load_text("m.yaml", "generate:\n  grid: [4, 4\nterms: []\n")
        assert str(e.value)

    def test_empty_file_is_refused(self):
        with pytest.raises(DetectionError) as e:
            load_text("m.yaml", "   \n\n")
        assert "empty" in str(e.value).lower()


class TestProgramShape:
    """Parsing is not the same as being a thermodynamic program. A file can be
    perfectly good YAML and still not be one, and the loader says which."""

    def test_yaml_without_variables_or_generate_is_named_as_not_a_program(self):
        with pytest.raises(DetectionError) as e:
            load_text("m.yaml", "title: my notes\nbody: hello\n")
        msg = str(e.value)
        assert "variables" in msg and "generate" in msg

    def test_a_list_at_top_level_is_refused(self):
        with pytest.raises(DetectionError) as e:
            load_text("m.yaml", "- one\n- two\n")
        assert "mapping" in str(e.value).lower()

    def test_generate_block_is_accepted(self):
        assert load_text("m.yaml", GRID_YAML).spec["generate"]["kind"] == "grid"

    def test_variables_block_is_accepted(self):
        text = "name: demo\nvariables:\n  a: {domain: binary}\nterms: []\n"
        assert "a" in load_text("m.yaml", text).spec["variables"]

    def test_generate_without_a_kind_is_caught_at_the_door(self):
        """Every generator reads `kind` first, so a block without one dies as a
        bare KeyError deep in the compiler. Catching it here turns a stack
        trace into a sentence."""
        with pytest.raises(DetectionError) as e:
            load_text("m.yaml", "generate:\n  grid: [8, 8]\n  weight: -1.0\nterms: []\n")
        assert "kind" in str(e.value)

    def test_generate_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(DetectionError) as e:
            load_text("m.yaml", "generate: [1, 2]\nterms: []\n")
        assert "mapping" in str(e.value).lower()

    def test_the_compilers_own_loader_gets_the_last_word(self):
        """`variable_domain: binary` is the obvious thing to write, and the
        compiler wants `variable_domain: {domain: binary}`. Left to the
        compiler the obvious version surfaces as `TypeError: string indices
        must be integers` after a long placement wait. The door delegates to
        the compiler's own loader so it lands at open time instead.

        This must not be an importorskip: a control that skips is not a
        control. The module header puts the compiler on the path.
        """
        import tsu_compiler.spec  # noqa: F401  -- fail loudly if unavailable
        bad = ("name: demo\ngenerate:\n  kind: grid\n  width: 4\n  height: 4\n"
               "  variable_domain: binary\nterms: []\n")
        with pytest.raises(DetectionError) as e:
            load_text("m.yaml", bad)
        assert "compiler cannot read it" in str(e.value)

    def test_a_missing_name_is_caught_at_the_door_too(self):
        """A spec needs a top-level `name`, and without one the compiler
        raises a bare KeyError. Delegation catches it without this door having
        to know the rule."""
        import tsu_compiler.spec  # noqa: F401
        with pytest.raises(DetectionError) as e:
            load_text("m.yaml", GRID_YAML.replace("name: demo\n", ""))
        assert "compiler cannot read it" in str(e.value)

    def test_a_well_formed_generate_block_passes(self):
        text = ("name: demo\ngenerate:\n  kind: grid\n  width: 8\n  height: 8\n"
                "  variable_domain: {domain: binary}\nterms: []\n")
        assert load_text("m.yaml", text).spec["generate"]["kind"] == "grid"


class TestYamlOutput:
    """Whatever came in, the editor shows YAML, because that is what the
    compiler's own load_spec reads. The conversion is part of the load so the
    reader can see and edit exactly what will be compiled."""

    def test_json_input_becomes_yaml_text(self):
        out = load_text("m.json", '{"name": "demo", "generate": {"kind": "grid", "width": 4, "height": 4, "variable_domain": {"domain": "binary"}}, "terms": []}')
        assert "generate:" in out.yaml
        assert "{" not in out.yaml.split("terms")[0]

    def test_yaml_input_is_passed_through_unchanged(self):
        """Round-tripping YAML through a parser destroys comments and ordering,
        and the reader wrote those on purpose."""
        text = "# my note\n" + GRID_YAML
        assert load_text("m.yaml", text).yaml == text

    def test_python_input_becomes_yaml_text(self):
        src = "SPEC = {'name': 'demo', 'generate': {'kind': 'grid', 'width': 2, 'height': 2, 'variable_domain': {'domain': 'binary'}}, 'terms': []}\n"
        assert "generate:" in load_text("model.py", src).yaml

    def test_converted_yaml_is_reparseable(self):
        out = load_text("m.json", '{"name": "demo", "generate": {"kind": "grid", "width": 4, "height": 4, "variable_domain": {"domain": "binary"}}, "terms": []}')
        assert load_text("again.yaml", out.yaml).spec == out.spec


class TestLoadedMessage:
    """What the reader sees: loaded <name>, and what it was read as."""

    def test_message_names_the_file(self):
        out = load_text("emergence_8x8.yaml", GRID_YAML)
        assert "emergence_8x8.yaml" in out.message
        assert out.message.lower().startswith("loaded")

    def test_message_names_a_converted_format(self):
        out = load_text("m.json", '{"name": "demo", "generate": {"kind": "grid", "width": 4, "height": 4, "variable_domain": {"domain": "binary"}}, "terms": []}')
        assert "json" in out.message.lower()

    def test_python_message_says_it_was_not_run(self):
        src = "SPEC = {'name': 'demo', 'generate': {'kind': 'grid', 'width': 2, 'height': 2, 'variable_domain': {'domain': 'binary'}}, 'terms': []}\n"
        out = load_text("model.py", src)
        assert "not" in out.message.lower() and "run" in out.message.lower()


class TestPythonRefusalReadability:
    """A refusal a reader cannot act on is barely better than a crash.

    `ast.literal_eval` raises with the offending node's full AST repr attached
    -- `DictComp(key=JoinedStr(values=[Constant(value='v', kind=None), ...`
    which buries a perfectly good sentence under a parser dump. Name the
    construct instead.
    """

    def test_a_dict_comprehension_is_named_in_plain_words(self):
        src = "SPEC = {f'v{i}': 1 for i in range(3)}\n"
        with pytest.raises(DetectionError) as e:
            load_text("m.py", src)
        msg = str(e.value)
        assert "comprehension" in msg
        assert "DictComp" not in msg, "the AST repr must not reach the reader"
        assert "Constant(" not in msg

    def test_a_list_comprehension_is_named(self):
        src = "SPEC = {'terms': [x for x in range(3)]}\n"
        with pytest.raises(DetectionError) as e:
            load_text("m.py", src)
        assert "comprehension" in str(e.value)

    def test_a_function_call_is_named_as_a_call(self):
        src = "SPEC = dict(a=1)\n"
        with pytest.raises(DetectionError) as e:
            load_text("m.py", src)
        assert "call" in str(e.value).lower()

    def test_a_reference_to_another_name_is_named(self):
        src = "base = {'a': 1}\nSPEC = base\n"
        with pytest.raises(DetectionError) as e:
            load_text("m.py", src)
        msg = str(e.value)
        assert "base" in msg, "name the variable it points at"

    def test_the_refusal_still_says_what_to_do_instead(self):
        src = "SPEC = {f'v{i}': 1 for i in range(3)}\n"
        with pytest.raises(DetectionError) as e:
            load_text("m.py", src)
        msg = str(e.value).lower()
        assert "yaml" in msg or "json" in msg
