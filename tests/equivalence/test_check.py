"""Tests for step 4: the differential equivalence check against the original.

The examples from the ticket are ``dafny``-marked tests with hand-written Dafny translations: a correct
palindrome translation passes, and a planted ``py_slice(s, 1, -2)`` bug fails with a concrete input. The
timeout, raises-versus-returns and tool-error edges run without Dafny, with the build and the sandbox stubbed.
"""

from __future__ import annotations

import importlib
import inspect
import io
import json
import re
import sys
import tokenize
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import pcc.equivalence.wrapper as wrapper_module
from pcc.core import (
    CheckKind,
    CheckRecord,
    Counterexample,
    FunctionRef,
    ToolError,
    Translation,
    TranslationPath,
    UnsupportedConstruct,
    source_hash,
)
from pcc.equivalence import check
from pcc.equivalence.check import SCHEMA, STEP, _model_edge_cases, _valid_call_source
from pcc.models import FakeClient
from pcc.sandbox import RunOutcome, run_python
from tests.helpers import function_ref

FIXTURES = Path(__file__).parent / "fixtures"

PALINDROME_SPEC = """predicate pre(s: string) { true }
predicate spec(s: string, result: bool) { result == (s == py_reversed(s)) }
"""

PALINDROME_CORRECT = """method is_palindrome(s: string) returns (result: bool)
  requires pre(s)
  ensures spec(s, result)
{
  return s == py_reversed(s);
}
"""

PALINDROME_BUGGY = """method is_palindrome(s: string) returns (result: bool)
  requires pre(s)
  ensures spec(s, result)
{
  return s == py_slice(s, 1, -2);
}
"""

HEAD_SPEC = """predicate pre(xs: seq<int>) { true }
predicate spec(xs: seq<int>, result: int) { result == py_index(xs, 0) }
"""

HEAD_RETURNS_WHERE_ORIGINAL_RAISES = """method head(xs: seq<int>) returns (result: int)
  requires pre(xs)
  ensures spec(xs, result)
{
  if |xs| == 0 {
    return -99;
  }
  return py_index(xs, 0);
}
"""


def load_fixture(name: str) -> tuple[str, str, str]:
    """The (module file name, qualname, source) of the ``tests/equivalence/fixtures`` entry ``name``."""
    return (f"{name}.py", name, (FIXTURES / f"{name}.py").read_text(encoding="utf-8"))


def function_at(function_name: str, repo_root: Path) -> FunctionRef:
    """Copy the fixture into ``repo_root`` and describe it as a ``FunctionRef``."""
    filename, _, source = load_fixture(function_name)
    (repo_root / filename).write_text(source, encoding="utf-8")
    match = re.search(r"^def (\w+)\(", source, flags=re.MULTILINE)
    assert match is not None, source
    qualname = match.group(1)
    lineno = next(i for i, line in enumerate(source.splitlines(), start=1) if line.startswith("def "))
    return FunctionRef(filename, qualname, lineno, source, source_hash(source))


def make_translation(function: FunctionRef, dafny_function: str, dafny_spec: str) -> Translation:
    """A transpiler-path translation with hand-written Dafny code and spec."""
    return Translation(function, TranslationPath.TRANSPILER, TranslationPath.TRANSPILER, dafny_function, dafny_spec)


def edge_answer(cases: list[str]) -> str:
    """A scripted ``edge_cases`` model answer."""
    return json.dumps({"edge_cases": cases})


check_module: ModuleType = importlib.import_module("pcc.equivalence.check")


def run_generated_test(mismatch: Counterexample, repo_root: Path) -> None:
    """Execute the generated pytest module of ``mismatch`` against the original; it must pass."""
    match = re.search(r"^def (test_\w+)\(\):", mismatch.test, flags=re.MULTILINE)
    assert match is not None, mismatch.test
    outcome = run_python(mismatch.test + f"\n{match.group(1)}()\n", cwd=repo_root)
    assert outcome.ok, outcome.stdout + outcome.stderr


@pytest.mark.dafny
def test_correct_palindrome_translation_passes(tmp_path: Path) -> None:
    """Ticket example 1: a correct palindrome translation passes the DIFFERENTIAL record."""
    function = function_at("palindrome", tmp_path)
    translation = make_translation(function, PALINDROME_CORRECT, PALINDROME_SPEC)
    client = FakeClient([edge_answer(["'aba'", "'abba'", "''", "'abc'"])])

    report = check(function, translation, tmp_path, inputs=100, symbolic_seconds=0.0, client=client, model="m")

    assert [record.kind for record in report.records] == [CheckKind.DIFFERENTIAL]
    assert report.records[0].passed is True
    assert report.records[0].detail == ""
    assert report.records[0].inputs_tried == report.records[0].inputs_total
    assert (report.records[0].inputs_tried or 0) > 0
    assert report.mismatch is None
    assert client.calls[0].step == "edge_cases"


@pytest.mark.dafny
def test_planted_slice_bug_fails_with_concrete_input(tmp_path: Path) -> None:
    """Ticket example 2: ``py_slice(s, 1, -2)`` instead of ``py_reversed(s)`` fails with a concrete input."""
    function = function_at("palindrome", tmp_path)
    translation = make_translation(function, PALINDROME_BUGGY, PALINDROME_SPEC)
    client = FakeClient([edge_answer([])])

    report = check(function, translation, tmp_path, inputs=50, symbolic_seconds=60.0, client=client, model="m")

    assert [record.kind for record in report.records] == [CheckKind.DIFFERENTIAL]
    assert report.records[0].passed is False
    assert report.mismatch is not None
    assert report.mismatch.expected.startswith("returns ")
    assert report.mismatch.actual.startswith("returns ")
    assert report.mismatch.expected != report.mismatch.actual
    assert report.mismatch.arguments in report.records[0].detail
    run_generated_test(report.mismatch, tmp_path)


@pytest.mark.dafny
def test_original_raises_and_translation_returns_is_a_mismatch(tmp_path: Path) -> None:
    """Ticket edge case: the original raises ``IndexError`` on ``[]`` while the translation returns ``-99``."""
    function = function_at("head", tmp_path)
    translation = make_translation(function, HEAD_RETURNS_WHERE_ORIGINAL_RAISES, HEAD_SPEC)
    client = FakeClient([edge_answer(["[]"])])

    report = check(function, translation, tmp_path, inputs=50, symbolic_seconds=0.0, client=client, model="m")

    assert report.records[0].passed is False
    assert report.mismatch is not None
    assert report.mismatch.arguments == "[]"
    assert report.mismatch.expected == "raises IndexError"
    assert report.mismatch.actual == "returns -99"
    run_generated_test(report.mismatch, tmp_path)


def _ok(stdout: str) -> RunOutcome:
    return RunOutcome(ok=True, returncode=0, stdout=stdout, stderr="", timed_out=False, seconds=0.0)


def _code_without_comments_or_strings(source: str) -> str:
    """Executable code of ``source`` without comments and string/docstring literals."""
    kept: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return "\n".join(kept)


def test_symbolic_seconds_writes_no_symbolic_record_and_runs_no_crosshair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``symbolic_seconds`` is ignored: one DIFFERENTIAL record, no SYMBOLIC record, no crosshair anywhere."""
    filename, qualname, source = load_fixture("adder")
    function = FunctionRef(filename, qualname, 4, source, source_hash(source))
    translation = make_translation(function, "method add() returns (r: int) { r := 0; }", "predicate pre() { true }")
    client = FakeClient([edge_answer(["1, 2"])])

    def fake_build(dafny_code: str, out_dir: Path) -> Path:
        assert "method add" in dafny_code
        return tmp_path / "m-py"

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        assert cwd == tmp_path
        if "PCC_EQUIV_INPUTS" in script:
            return _ok('PCC_EQUIV_INPUTS {"inputs": ["1, 2", "0, -1"]}\n')
        assert "PCC_EQUIV_DONE" in script
        return _ok('PCC_EQUIV_DONE {"tried": 2, "total": 2, "mismatch": null}\n')

    modules_before = {name for name in sys.modules if "crosshair" in name}
    monkeypatch.setattr(check_module, "build_python", fake_build)
    monkeypatch.setattr(check_module, "run_python", fake_run)

    report = check(function, translation, tmp_path, inputs=10, symbolic_seconds=60.0, client=client, model="m")

    assert report.records == (CheckRecord(CheckKind.DIFFERENTIAL, True, "", 2, 2),)
    assert report.mismatch is None
    assert len(client.requests) == 1 and client.requests[0][2] == "edge_cases"
    assert client.calls[0].schema == SCHEMA
    assert STEP == "edge_cases"
    assert {name for name in sys.modules if "crosshair" in name} == modules_before
    combined = inspect.getsource(check_module) + inspect.getsource(wrapper_module)
    assert "crosshair" not in _code_without_comments_or_strings(combined).lower()


def test_translation_timeout_is_a_timeout_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Both sides hanging on the same input fails without a mismatch instead of identical outcomes."""
    function = function_at("adder", tmp_path)
    translation = make_translation(function, "method add() returns (r: int) { r := 0; }", "predicate pre() { true }")
    client = FakeClient([edge_answer([])])

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        if "PCC_EQUIV_INPUTS" in script:
            return _ok('PCC_EQUIV_INPUTS {"inputs": ["1, 2"]}\n')
        return RunOutcome(
            ok=False, returncode=1, stdout='PCC_EQUIV_PROGRESS {"idx": 0}\n', stderr="", timed_out=True, seconds=0.0
        )

    def always_timeout(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        return RunOutcome(ok=False, returncode=1, stdout="", stderr="", timed_out=True, seconds=0.0)

    monkeypatch.setattr(check_module, "build_python", lambda code, out: tmp_path / "m-py")
    monkeypatch.setattr(check_module, "run_python", fake_run)
    monkeypatch.setattr("pcc.sandbox.run_python", always_timeout)

    report = check(function, translation, tmp_path, inputs=10, symbolic_seconds=0.0, client=client, model="m")

    assert report.records[0].kind is CheckKind.DIFFERENTIAL
    assert report.records[0].passed is False
    assert "timeout" in report.records[0].detail
    assert "both sides" in report.records[0].detail
    assert report.mismatch is None


def test_translation_timeout_with_answering_original_replays_original(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hanging translation with an answering original reports the original outcome and a runnable test."""
    function = function_at("adder", tmp_path)
    translation = make_translation(function, "method add() returns (r: int) { r := 0; }", "predicate pre() { true }")
    client = FakeClient([edge_answer([])])

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        if "PCC_EQUIV_INPUTS" in script:
            return _ok('PCC_EQUIV_INPUTS {"inputs": ["1, 2"]}\n')
        return RunOutcome(
            ok=False, returncode=1, stdout='PCC_EQUIV_PROGRESS {"idx": 0}\n', stderr="", timed_out=True, seconds=0.0
        )

    def pinpoint_or_probe(
        script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024
    ) -> RunOutcome:
        if "PCC_EQUIV_ORIG" in script:
            return _ok('PCC_EQUIV_ORIG {"kind": "return", "payload": "3"}\n')
        return RunOutcome(ok=False, returncode=1, stdout="", stderr="", timed_out=True, seconds=0.0)

    monkeypatch.setattr(check_module, "build_python", lambda code, out: tmp_path / "m-py")
    monkeypatch.setattr(check_module, "run_python", fake_run)
    monkeypatch.setattr("pcc.sandbox.run_python", pinpoint_or_probe)

    report = check(function, translation, tmp_path, inputs=10, symbolic_seconds=0.0, client=client, model="m")

    assert report.records[0].passed is False
    assert report.mismatch is not None
    assert report.mismatch.arguments == "1, 2"
    assert report.mismatch.expected == "returns 3"
    assert report.mismatch.actual == "timeout"
    run_generated_test(report.mismatch, tmp_path)


def test_precondition_never_holds_is_not_a_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No input with ``pre`` holding: a failed DIFFERENTIAL record without a mismatch."""
    filename, qualname, source = load_fixture("adder")
    function = FunctionRef(filename, qualname, 4, source, source_hash(source))
    translation = make_translation(function, "method add() returns (r: int) { r := 0; }", "predicate pre() { true }")
    client = FakeClient([edge_answer([])])

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        if "PCC_EQUIV_INPUTS" in script:
            return _ok('PCC_EQUIV_INPUTS {"inputs": ["1, 2"]}\n')
        return _ok('PCC_EQUIV_DONE {"tried": 0, "total": 7, "mismatch": null}\n')

    monkeypatch.setattr(check_module, "build_python", lambda code, out: tmp_path / "m-py")
    monkeypatch.setattr(check_module, "run_python", fake_run)

    report = check(function, translation, tmp_path, inputs=10, symbolic_seconds=0.0, client=client, model="m")

    assert report.records == (CheckRecord(CheckKind.DIFFERENTIAL, False, "precondition never holds", 0, 7),)
    assert report.mismatch is None


@pytest.mark.parametrize(
    "collect_stdout,collect_timed_out",
    [
        ("PCC_EQUIV_INPUTS not json\n", False),
        ("", True),
    ],
)
def test_broken_collection_run_raises_tool_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, collect_stdout: str, collect_timed_out: bool
) -> None:
    """A collection run with no parseable marker is a ``ToolError``, never mistaken for a pass."""
    filename, qualname, source = load_fixture("adder")
    function = FunctionRef(filename, qualname, 4, source, source_hash(source))
    translation = make_translation(function, "method add() returns (r: int) { r := 0; }", "predicate pre() { true }")

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        return RunOutcome(
            ok=False, returncode=1, stdout=collect_stdout, stderr="boom", timed_out=collect_timed_out, seconds=0.0
        )

    monkeypatch.setattr(check_module, "build_python", lambda code, out: tmp_path / "m-py")
    monkeypatch.setattr(check_module, "run_python", fake_run)

    with pytest.raises(ToolError):
        check(function, translation, tmp_path, inputs=10, symbolic_seconds=0.0, client=FakeClient([]), model="m")


@pytest.mark.parametrize(
    "done_payload",
    [
        '{"tried": "two", "total": 2, "mismatch": null}',
        '{"tried": 1, "total": 2, "mismatch": {"arguments": 42}}',
        '{"tried": 1, "total": 2, "mismatch": []}',
    ],
)
def test_malformed_evaluation_result_raises_tool_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, done_payload: str
) -> None:
    """A malformed ``PCC_EQUIV_DONE`` line is a ``ToolError``, never mistaken for a pass."""
    filename, qualname, source = load_fixture("adder")
    function = FunctionRef(filename, qualname, 4, source, source_hash(source))
    translation = make_translation(function, "method add() returns (r: int) { r := 0; }", "predicate pre() { true }")
    client = FakeClient([edge_answer([])])

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        if "PCC_EQUIV_INPUTS" in script:
            return _ok('PCC_EQUIV_INPUTS {"inputs": ["1, 2"]}\n')
        return _ok(f"PCC_EQUIV_DONE {done_payload}\n")

    monkeypatch.setattr(check_module, "build_python", lambda code, out: tmp_path / "m-py")
    monkeypatch.setattr(check_module, "run_python", fake_run)

    with pytest.raises(ToolError):
        check(function, translation, tmp_path, inputs=10, symbolic_seconds=0.0, client=client, model="m")


def test_unconvertible_hint_raises_unsupported_construct(tmp_path: Path) -> None:
    """``float`` has a Hypothesis strategy but no converter, so it never becomes a false mismatch."""
    function = function_ref(source="def f(x: float) -> float:\n    return x\n", path="f.py", qualname="f")
    translation = make_translation(function, "method f() returns (r: int) { r := 0; }", "predicate pre() { true }")

    with pytest.raises(UnsupportedConstruct):
        check(function, translation, tmp_path, inputs=10, symbolic_seconds=0.0, client=FakeClient([]), model="m")


def test_model_edge_cases_drops_invalid_entries() -> None:
    """Only parseable call-argument sources with known keywords and fitting arity become edge cases."""
    function = function_ref(
        source="def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n", path="palindrome.py", qualname="s"
    )
    client = FakeClient([edge_answer(["'aba'", "x[", "*rest", "t=1", "'a', 'b'", "'aba'", "  ", "s='aa'"])])

    assert _model_edge_cases(function, client=client, model="m") == ("'aba'", "s='aa'")
    assert client.requests[0][2] == "edge_cases"


def test_model_edge_cases_are_capped_at_twenty() -> None:
    """More than 20 valid model edge cases are cut down to the first 20."""
    function = function_ref(
        source="def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n", path="palindrome.py", qualname="s"
    )
    cases = [f"'case{i}'" for i in range(25)]
    client = FakeClient([edge_answer(cases)])

    assert _model_edge_cases(function, client=client, model="m") == tuple(cases[:20])


@pytest.mark.parametrize("answer", ["not json", "[1, 2]", '{"other": []}', '{"edge_cases": "nope"}'])
def test_model_edge_cases_malformed_answer_contributes_nothing(answer: str) -> None:
    """A malformed model answer contributes zero edge cases instead of failing the check."""
    function = function_ref(
        source="def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n", path="palindrome.py", qualname="s"
    )

    assert _model_edge_cases(function, client=FakeClient([answer]), model="m") == ()


def test_valid_call_source_rejects_starred_and_unknown_keywords() -> None:
    """``*args`` and unknown keywords never become edge cases; defaults allow fewer positional arguments."""
    assert _valid_call_source("'a'", 1, {"s"}) is True
    assert _valid_call_source("s='a'", 1, {"s"}) is True
    assert _valid_call_source("*rest", 1, {"s"}) is False
    assert _valid_call_source("t='a'", 1, {"s"}) is False
    assert _valid_call_source("'a', 'b'", 1, {"s"}) is False
    assert _valid_call_source("x[", 1, {"s"}) is False


def test_eval_script_applies_missing_defaults() -> None:
    """Inputs omitting defaulted parameters bind defaults instead of being skipped."""
    function = function_ref(
        source="def greet(name: str, greeting: str = 'hi') -> str:\n    return greeting + name\n",
        path="greet.py",
        qualname="greet",
    )

    script: str = check_module._eval_script(function, "M", "pass", ["'bob'"], "pre")

    assert "apply_defaults" in script
    start = script.index("def _ordered(")
    end = script.index("_pre = getattr(")
    ordered_source = script[start:end]

    def greet(name: str, greeting: str = "hi") -> str:
        return greeting + name

    namespace: dict[str, Any] = {"inspect": inspect, "target": greet, "_NAMES": ["name", "greeting"]}
    exec(compile(ordered_source, "<ordered>", "exec"), namespace)
    ordered: Any = namespace["_ordered"]
    assert callable(ordered)
    assert ordered(("bob",), {}) == ("bob", "hi")
    assert ordered((), {}) is None
    assert ordered(("a", "b", "c"), {}) is None


def test_wrapper_keeps_exact_parameters_and_hints() -> None:
    """``translated`` copies positional-only, defaulted and keyword-only parameters with their hints."""
    function = function_ref(
        source="def f(a: int, /, b: str = 'hi', *, c: bool = True) -> int:\n    return 1\n",
        path="f.py",
        qualname="f",
    )

    source = wrapper_module.wrapper_source(function, "PCC_f__f_abcdef12")

    assert "def translated(a: int, /, b: str = 'hi', *, c: bool = True) -> int:" in source
    assert "to_dafny(a, int)" in source
    assert "to_dafny(b, str)" in source
    assert "to_dafny(c, bool)" in source
    assert "return to_python(_dafny_result, int)" in source
    assert "_dafny_module.default__.f(" in source
    assert "def to_dafny(" in source and "def to_python(" in source


def test_wrapper_rejects_missing_hints_and_variadics() -> None:
    """A missing hint or ``*args``/``**kwargs`` raises ``UnsupportedConstruct`` for the wrapper too."""
    untyped = function_ref(source="def f(a) -> int:\n    return 1\n", path="f.py", qualname="f")
    variadic = function_ref(source="def f(*args: int) -> int:\n    return 1\n", path="f.py", qualname="f")
    untyped_return = function_ref(source="def f(a: int):\n    return 1\n", path="f.py", qualname="f")

    with pytest.raises(UnsupportedConstruct):
        wrapper_module.wrapper_source(untyped, "M")
    with pytest.raises(UnsupportedConstruct):
        wrapper_module.wrapper_source(variadic, "M")
    with pytest.raises(UnsupportedConstruct):
        wrapper_module.wrapper_source(untyped_return, "M")
