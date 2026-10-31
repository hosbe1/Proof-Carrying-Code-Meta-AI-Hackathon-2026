"""Tests for step 4, stretch: the symbolic equivalence check with crosshair diffbehavior (#75).

The symbolic phase runs after the differential test whenever ``symbolic_seconds > 0``: one importable sandbox
module defines ``original`` and ``translated`` with the original's exact typed signature, both guarded by the
compiled Dafny ``pre``, and ``crosshair diffbehavior`` compares them. The examples from the ticket are
``dafny``-marked tests with hand-written translations: correct ``inc`` and palindrome translations pass, a planted
``n == 123457`` bug the differential test misses is found by the symbolic search, and ``symbolic_seconds=0``
writes no SYMBOLIC record. Stubbed runs (no Dafny, no crosshair) pin the timeout, crash, precedence and
``ToolError`` edges, plus crosshair's exit-code and output contract against the installed package.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import pcc.equivalence.wrapper as wrapper_module
from pcc.core import CheckKind, CheckRecord, FunctionRef, ToolError, source_hash
from pcc.equivalence import check
from pcc.models import FakeClient
from pcc.sandbox import RunOutcome, run_python
from tests.equivalence.test_check import edge_answer, function_at, make_translation, run_generated_test
from tests.helpers import function_ref

check_module: ModuleType = import_module("pcc.equivalence.check")

FIXTURES = Path(__file__).parent / "fixtures"

INC_SPEC = """predicate pre(n: int) { true }
predicate spec(n: int, result: int) { result == n + 1 }
"""

INC_CORRECT = """method inc(n: int) returns (result: int)
  requires pre(n)
  ensures spec(n, result)
{
  return n + 1;
}
"""

INC_BUGGY = """method inc(n: int) returns (result: int)
  requires pre(n)
  ensures spec(n, result)
{
  if n == 123457 {
    return 0;
  }
  return n + 1;
}
"""

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

RunStub = Callable[..., RunOutcome]


def _ok(stdout: str) -> RunOutcome:
    return RunOutcome(ok=True, returncode=0, stdout=stdout, stderr="", timed_out=False, seconds=0.0)


def _diff_stub(tmp_path: Path, mismatch: Mapping[str, Any] | None = None) -> RunStub:
    """A ``run_python`` stub answering the differential scripts; anything else is a test failure."""
    done = {"tried": 3, "total": 3, "mismatch": mismatch}

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        assert cwd == tmp_path
        if "PCC_EQUIV_INPUTS" in script:
            return _ok('PCC_EQUIV_INPUTS {"inputs": ["1", "2", "-3"]}\n')
        if "PCC_EQUIV_DONE" in script:
            return _ok(f"PCC_EQUIV_DONE {json.dumps(done)}\n")
        raise AssertionError(f"unexpected sandbox script: {script[:200]}")

    return fake_run


def _with_symbolic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, symbolic: RunStub, mismatch: Mapping[str, Any] | None = None
) -> None:
    """Answer differential scripts with a stub and ``crosshair`` scripts with ``symbolic``."""
    differential = _diff_stub(tmp_path, mismatch)

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        if "crosshair" in script.lower():
            return symbolic(script, cwd=cwd, timeout_seconds=timeout_seconds, memory_mb=memory_mb)
        return differential(script, cwd=cwd, timeout_seconds=timeout_seconds, memory_mb=memory_mb)

    monkeypatch.setattr(check_module, "build_python", lambda code, out: tmp_path / "m-py")
    monkeypatch.setattr(check_module, "run_python", fake_run)


def test_symbolic_module_keeps_exact_signature_and_pre_guard() -> None:
    """The symbolic module defines both sides with the original hints and a ``pre`` guard returning ``None``."""
    function = function_ref(source="def inc(n: int) -> int:\n    return n + 1\n", path="inc.py", qualname="inc")

    source = wrapper_module.symbolic_module_source(function, "PCC_inc__inc_abcdef12")

    compile(source, "pcc_symbolic.py", "exec")
    assert "def original(n: int) -> int:" in source
    assert "def translated(n: int) -> int:" in source
    assert "def _translated_impl(n: int) -> int:" in source
    assert "return None" in source
    assert "default__" in source
    assert "target(" in source
    assert "_translated_impl(" in source


def test_differential_wrapper_mentions_no_crosshair() -> None:
    """The #37 differential wrapper stays free of crosshair: the symbolic layer lives beside it."""
    function = function_ref(source="def inc(n: int) -> int:\n    return n + 1\n", path="inc.py", qualname="inc")

    assert "crosshair" not in wrapper_module.wrapper_source(function, "PCC_inc__inc_abcdef12").lower()


@pytest.mark.dafny
def test_correct_inc_translation_symbolic_record_passes(tmp_path: Path) -> None:
    """Ticket example 1 (inc shape): a correct translation passes the SYMBOLIC record with no mismatch."""
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_CORRECT, INC_SPEC)
    client = FakeClient([edge_answer([])])

    report = check(function, translation, tmp_path, inputs=50, symbolic_seconds=30.0, client=client, model="m")

    assert [record.kind for record in report.records] == [CheckKind.DIFFERENTIAL, CheckKind.SYMBOLIC]
    assert report.records[0].passed is True
    symbolic = report.records[1]
    assert symbolic.passed is True
    assert symbolic.detail == "no differences found"
    assert symbolic.inputs_tried is None and symbolic.inputs_total is None
    assert report.mismatch is None


@pytest.mark.dafny
def test_correct_palindrome_translation_symbolic_record_passes(tmp_path: Path) -> None:
    """Ticket example 1: a correct palindrome translation passes the SYMBOLIC record with no mismatch."""
    function = function_at("palindrome", tmp_path)
    translation = make_translation(function, PALINDROME_CORRECT, PALINDROME_SPEC)
    client = FakeClient([edge_answer([])])

    report = check(function, translation, tmp_path, inputs=20, symbolic_seconds=10.0, client=client, model="m")

    assert [record.kind for record in report.records] == [CheckKind.DIFFERENTIAL, CheckKind.SYMBOLIC]
    assert report.records[0].passed is True
    symbolic = report.records[1]
    assert symbolic.passed is True
    assert symbolic.detail == "no differences found"
    assert symbolic.inputs_tried is None and symbolic.inputs_total is None
    assert report.mismatch is None


@pytest.mark.dafny
def test_planted_bug_missed_by_differential_is_found_symbolically(tmp_path: Path) -> None:
    """Ticket example 2: the differential test misses ``n == 123457`` and the symbolic search finds it."""
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_BUGGY, INC_SPEC)
    client = FakeClient([edge_answer([])])

    report = check(function, translation, tmp_path, inputs=50, symbolic_seconds=30.0, client=client, model="m")

    assert [record.kind for record in report.records] == [CheckKind.DIFFERENTIAL, CheckKind.SYMBOLIC]
    assert report.records[0].passed is True
    symbolic = report.records[1]
    assert symbolic.passed is False
    assert "123457" in symbolic.detail
    assert symbolic.inputs_tried is None and symbolic.inputs_total is None
    assert report.mismatch is not None
    assert "123457" in report.mismatch.arguments
    assert report.mismatch.expected != report.mismatch.actual
    run_generated_test(report.mismatch, tmp_path)


def test_symbolic_seconds_zero_writes_no_symbolic_record_and_runs_no_crosshair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ticket example 3: ``symbolic_seconds=0`` keeps the #37 differential-only result; crosshair never runs."""
    filename = "inc.py"
    source = (FIXTURES / filename).read_text(encoding="utf-8")
    function = FunctionRef(filename, "inc", 4, source, source_hash(source))
    (tmp_path / filename).write_text(source, encoding="utf-8")
    translation = make_translation(function, INC_CORRECT, INC_SPEC)
    client = FakeClient([edge_answer([])])
    seen: list[str] = []

    def fake_run(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        seen.append(script)
        if "PCC_EQUIV_INPUTS" in script:
            return _ok('PCC_EQUIV_INPUTS {"inputs": ["1", "2"]}\n')
        assert "PCC_EQUIV_DONE" in script
        return _ok('PCC_EQUIV_DONE {"tried": 2, "total": 2, "mismatch": null}\n')

    modules_before = {name for name in sys.modules if "crosshair" in name}
    monkeypatch.setattr(check_module, "build_python", lambda code, out: tmp_path / "m-py")
    monkeypatch.setattr(check_module, "run_python", fake_run)

    report = check(function, translation, tmp_path, inputs=10, symbolic_seconds=0.0, client=client, model="m")

    assert report.records == (CheckRecord(CheckKind.DIFFERENTIAL, True, "", 2, 2),)
    assert report.mismatch is None
    assert seen and all("crosshair" not in script.lower() for script in seen)
    assert {name for name in sys.modules if "crosshair" in name} == modules_before


def test_symbolic_argv_matches_the_documented_diffbehavior_call() -> None:
    """The prescribed argv: ``diffbehavior`` with ``--per_condition_timeout`` and the two entry points."""
    assert check_module._symbolic_argv(20.0) == [
        "crosshair",
        "diffbehavior",
        "--per_condition_timeout",
        "20.0",
        "pcc_symbolic.original",
        "pcc_symbolic.translated",
    ]


def test_crosshair_exit_contract_pinned_against_installed_package(tmp_path: Path) -> None:
    """Exit 0 with ``No differences found.`` on equal functions, exit 1 with ``Given: (...)`` on a difference."""
    pair = (
        "def original(n: int) -> int:\n    return n + 1\n"
        "def translated(n: int) -> int:\n    return n + 1\n"
        "def other_original(n: int) -> int:\n    return n + 1\n"
        "def other_translated(n: int) -> int:\n    return 0 if n == 123457 else n + 1\n"
    )
    (tmp_path / "pcc_symbolic.py").write_text(pair, encoding="utf-8")
    runner = (
        "import sys\n"
        "sys.argv = ['crosshair', 'diffbehavior', '--per_condition_timeout', '10', "
        "'pcc_symbolic.{first}', 'pcc_symbolic.{second}']\n"
        "import crosshair.main\ncrosshair.main.main()\n"
    )

    same_outcome = run_python(runner.format(first="original", second="translated"), cwd=tmp_path, timeout_seconds=120.0)
    assert same_outcome.returncode == 0
    assert "No differences found." in same_outcome.stdout

    diff_outcome = run_python(
        runner.format(first="other_original", second="other_translated"), cwd=tmp_path, timeout_seconds=120.0
    )
    assert diff_outcome.returncode == 1
    assert "Given: (" in diff_outcome.stdout
    assert "123457" in diff_outcome.stdout


def test_symbolic_timeout_passes_within_budget_detail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A sandbox wall timeout is a pass: no difference was found within the budget."""
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_CORRECT, INC_SPEC)

    def symbolic(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        assert timeout_seconds == 45.0
        return RunOutcome(ok=False, returncode=1, stdout="", stderr="", timed_out=True, seconds=45.0)

    _with_symbolic(monkeypatch, tmp_path, symbolic)

    report = check(
        function,
        translation,
        tmp_path,
        inputs=10,
        symbolic_seconds=15.0,
        client=FakeClient([edge_answer([])]),
        model="m",
    )

    assert [record.kind for record in report.records] == [CheckKind.DIFFERENTIAL, CheckKind.SYMBOLIC]
    symbolic_record = report.records[1]
    assert symbolic_record.passed is True
    assert symbolic_record.detail == "no differences found within 15 s"
    assert report.mismatch is None


def test_symbolic_crash_fails_with_last_stderr_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exit 2 is a crash: the record fails with the last nonblank stderr line and sets no mismatch."""
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_CORRECT, INC_SPEC)

    def symbolic(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        return RunOutcome(
            ok=False,
            returncode=2,
            stdout="",
            stderr="worker crashed\nUnable to load it: boom\n\n",
            timed_out=False,
            seconds=1.0,
        )

    _with_symbolic(monkeypatch, tmp_path, symbolic)

    report = check(
        function,
        translation,
        tmp_path,
        inputs=10,
        symbolic_seconds=15.0,
        client=FakeClient([edge_answer([])]),
        model="m",
    )

    symbolic_record = report.records[1]
    assert symbolic_record.passed is False
    assert symbolic_record.detail == "Unable to load it: boom"
    assert report.mismatch is None


def test_symbolic_crash_with_blank_stderr_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A crash with blank stderr fails with the ``crosshair failed`` fallback."""
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_CORRECT, INC_SPEC)

    def symbolic(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        return RunOutcome(ok=False, returncode=2, stdout="garbage", stderr="   \n", timed_out=False, seconds=1.0)

    _with_symbolic(monkeypatch, tmp_path, symbolic)

    report = check(
        function,
        translation,
        tmp_path,
        inputs=10,
        symbolic_seconds=15.0,
        client=FakeClient([edge_answer([])]),
        model="m",
    )

    assert report.records[1].passed is False
    assert report.records[1].detail == "crosshair failed"
    assert report.mismatch is None


def test_symbolic_runs_despite_differential_mismatch_and_keeps_it_primary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SYMBOLIC record is present even after a differential mismatch, which stays the primary mismatch."""
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_CORRECT, INC_SPEC)
    diff_mismatch = {"arguments": "5", "orig": ["return", "6"], "trans": ["return", "7"]}

    def symbolic(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        return RunOutcome(
            ok=False, returncode=1, stdout="Given: (n=123457),\n", stderr="", timed_out=False, seconds=1.0
        )

    _with_symbolic(monkeypatch, tmp_path, symbolic, diff_mismatch)

    report = check(
        function,
        translation,
        tmp_path,
        inputs=10,
        symbolic_seconds=15.0,
        client=FakeClient([edge_answer([])]),
        model="m",
    )

    assert [record.kind for record in report.records] == [CheckKind.DIFFERENTIAL, CheckKind.SYMBOLIC]
    assert report.records[0].passed is False
    assert report.records[1].passed is False
    assert "123457" in report.records[1].detail
    assert report.mismatch is not None
    assert report.mismatch.arguments == "5"


def test_symbolic_difference_fills_absent_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no differential mismatch, the symbolic input becomes the report mismatch with a runnable test."""
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_CORRECT, INC_SPEC)

    def symbolic(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        return RunOutcome(
            ok=False,
            returncode=1,
            stdout=(
                "Given: (n=123457),\n"
                "    pcc_symbolic.original : returns 123458\n"
                "  pcc_symbolic.translated : returns 0\n"
            ),
            stderr="",
            timed_out=False,
            seconds=1.0,
        )

    _with_symbolic(monkeypatch, tmp_path, symbolic)

    report = check(
        function,
        translation,
        tmp_path,
        inputs=10,
        symbolic_seconds=15.0,
        client=FakeClient([edge_answer([])]),
        model="m",
    )

    assert report.records[1].passed is False
    assert report.mismatch is not None
    assert report.mismatch.arguments == "n=123457"
    run_generated_test(report.mismatch, tmp_path)


def test_symbolic_tool_error_escapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A ``ToolError`` from the symbolic sandbox run escapes instead of becoming a record."""
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_CORRECT, INC_SPEC)

    def symbolic(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        raise ToolError("sandbox exploded")

    _with_symbolic(monkeypatch, tmp_path, symbolic)

    with pytest.raises(ToolError):
        check(
            function,
            translation,
            tmp_path,
            inputs=10,
            symbolic_seconds=15.0,
            client=FakeClient([edge_answer([])]),
            model="m",
        )


def test_symbolic_record_carries_no_input_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CrossHair iterations are not the differential input count: both fields stay ``None``."""
    assert CheckRecord(CheckKind.SYMBOLIC, True, "no differences found").inputs_tried is None
    function = function_at("inc", tmp_path)
    translation = make_translation(function, INC_CORRECT, INC_SPEC)

    def symbolic(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
        return _ok("No differences found. (attempted 2 iterations)\n")

    _with_symbolic(monkeypatch, tmp_path, symbolic)

    report = check(
        function,
        translation,
        tmp_path,
        inputs=10,
        symbolic_seconds=15.0,
        client=FakeClient([edge_answer([])]),
        model="m",
    )

    symbolic_record = report.records[1]
    assert symbolic_record.kind is CheckKind.SYMBOLIC
    assert symbolic_record.inputs_tried is None and symbolic_record.inputs_total is None
