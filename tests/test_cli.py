"""Tests for ``pcc-check run`` (issue #40).

Every example and edge case of the ticket runs against a temporary repository with the pipeline
monkeypatched; only ``discover``, the certificate store and the report run for real.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from pcc import models, pipeline, project, verifier
from pcc.certificate import from_result, write
from pcc.cli import main
from pcc.config import Config
from pcc.core import (
    CheckKind,
    CheckRecord,
    Context,
    Counterexample,
    EquivalenceReport,
    Evidence,
    FunctionRef,
    Intent,
    ProofOutcome,
    Result,
    SecondOpinion,
    SourceKind,
    Spec,
    State,
    ToolError,
    Translation,
    serialize,
)
from pcc.verifier.dafny import VerifyOutcome
from tests import helpers as h

REPO_SOURCE = "def add(a: int, b: int) -> int:\n    return a + b\n"
REL_PATH = "pkg/mod.py"


def _write_repo(root: Path, source: str = REPO_SOURCE) -> Path:
    target = root / REL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def _discover(root: Path) -> FunctionRef:
    refs = project.discover(root, [REL_PATH])
    assert len(refs) == 1
    return refs[0]


def _proven_result(function: FunctionRef, *, reason: str = "verified") -> Result:
    return Result(
        function,
        State.PROVEN,
        reason,
        h.intent(function),
        h.spec(function),
        h.translation(function),
        (),
        None,
        h.proof(),
    )


def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "build_client", lambda config: models.FakeClient([]))
    monkeypatch.setattr(models, "available_stages", lambda config, router: config)


def _patch_affected_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(project, "affected", lambda root, functions: tuple(functions))


def _patch_run_many(
    monkeypatch: pytest.MonkeyPatch,
    results: Sequence[Result] | None = None,
    *,
    error: BaseException | None = None,
    calls: list[tuple[FunctionRef, ...]] | None = None,
    seen_callees: list[Mapping[FunctionRef, Result]] | None = None,
) -> None:
    def fake(
        functions: Sequence[FunctionRef],
        repo_root: Path,
        *,
        client: object,
        config: Config,
        callees: Mapping[FunctionRef, Result] | None = None,
    ) -> tuple[Result, ...]:
        if calls is not None:
            calls.append(tuple(functions))
        if seen_callees is not None:
            seen_callees.append(dict(callees or {}))
        if error is not None:
            raise error
        return tuple(results or ())

    monkeypatch.setattr(pipeline, "run_many", fake)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
) -> int:
    monkeypatch.chdir(tmp_path)
    return main(argv)


def _default_argv(tmp_path: Path) -> list[str]:
    proofs = tmp_path / "proofs"
    return [
        "run",
        REL_PATH,
        "--proofs-dir",
        str(proofs),
        "--report",
        str(tmp_path / "report.md"),
        "--json",
        str(tmp_path / "results.json"),
    ]


def _read_results(tmp_path: Path) -> list[Result]:
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return [serialize.from_data(Result, entry) for entry in data]


def test_version_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out == f"{version('pcc')}\n"


def test_help_lists_all_run_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in ("paths", "--base", "--proofs-dir", "--report", "--json"):
        assert option in out


def test_run_writes_certificate_for_proven_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    function = _discover(tmp_path)
    _patch_run_many(monkeypatch, [_proven_result(function)])
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 0
    written = tmp_path / "proofs" / "pkg" / "mod" / "add.dfy"
    assert written.is_file()
    assert (tmp_path / "report.md").is_file()
    results = _read_results(tmp_path)
    assert len(results) == 1
    assert results[0].state is State.PROVEN
    assert capsys.readouterr().out.strip() == "1 proven"


def test_run_counterexample_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    function = _discover(tmp_path)
    found = Result(
        function,
        State.COUNTEREXAMPLE,
        "breaks spec",
        h.intent(function),
        h.spec(function),
        h.translation(function),
        (),
        h.counterexample(),
    )
    _patch_run_many(monkeypatch, [found])
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 1


def test_run_tool_error_exits_2_with_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    _patch_run_many(monkeypatch, error=ToolError("model API down"))
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 2
    assert "model API down" in capsys.readouterr().err


def test_run_open_result_without_certificate_exits_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    function = _discover(tmp_path)
    _patch_run_many(monkeypatch, [Result(function, State.OPEN, "budget")])
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 0
    assert capsys.readouterr().out.strip() == "1 open"


def test_run_valid_certificate_skips_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    function = _discover(tmp_path)
    proofs = tmp_path / "proofs"
    write(from_result(_proven_result(function), Config()), proofs)
    calls: list[tuple[FunctionRef, ...]] = []
    _patch_run_many(monkeypatch, [Result(function, State.OPEN, "budget")], calls=calls)
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 0
    assert calls == []
    results = _read_results(tmp_path)
    assert len(results) == 1
    assert results[0].state is State.PROVEN
    assert results[0].reason == "certificate"
    assert results[0].proof is not None and results[0].proof.stage == "certificate"
    assert capsys.readouterr().out.strip() == "1 proven"


def test_run_runtime_error_exits_2_with_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    _patch_run_many(monkeypatch, error=RuntimeError("kaboom"))
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 2
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert "kaboom" in err


def test_run_pylib_change_with_failed_reverification_is_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    function = _discover(tmp_path)
    proofs = tmp_path / "proofs"
    write(from_result(_proven_result(function), Config()), proofs)
    sidecar = tmp_path / "proofs" / "pkg" / "mod" / "add.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["pylib_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    verify_calls: list[str] = []

    def fake_verify(dafny_code: str, **kwargs: object) -> VerifyOutcome:
        verify_calls.append(dafny_code)
        return VerifyOutcome(False, (), 0.0)

    monkeypatch.setattr(verifier, "verify", fake_verify)
    _patch_run_many(monkeypatch, [Result(function, State.OPEN, "no proof found")])
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 1
    assert len(verify_calls) == 1
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Stale certificates" in report
    assert "pkg/mod.py::add" in report


def test_run_no_changed_functions_reports_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_models(monkeypatch)
    monkeypatch.setattr(project, "changed", lambda root, base: ())
    calls: list[tuple[FunctionRef, ...]] = []
    _patch_run_many(monkeypatch, [], calls=calls)
    argv = [
        "run",
        "--proofs-dir",
        str(tmp_path / "proofs"),
        "--report",
        str(tmp_path / "report.md"),
        "--json",
        str(tmp_path / "results.json"),
    ]
    assert _run(monkeypatch, tmp_path, argv) == 0
    assert calls == []
    assert "No functions checked." in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out.strip() == "No functions checked."


def test_run_stale_certificate_with_open_result_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    function = _discover(tmp_path)
    proofs = tmp_path / "proofs"
    write(from_result(_proven_result(function), Config()), proofs)
    (tmp_path / REL_PATH).write_text("def add(a: int, b: int) -> int:\n    return a - b\n", encoding="utf-8")
    stale_function = _discover(tmp_path)
    _patch_run_many(monkeypatch, [Result(stale_function, State.OPEN, "no proof found")])
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 1
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "pkg/mod.py::add" in report


def test_run_pylib_change_with_successful_reverification_refreshes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    function = _discover(tmp_path)
    proofs = tmp_path / "proofs"
    write(from_result(_proven_result(function), Config()), proofs)
    sidecar = tmp_path / "proofs" / "pkg" / "mod" / "add.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["pylib_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    verify_calls: list[str] = []

    def fake_verify(dafny_code: str, **kwargs: object) -> VerifyOutcome:
        verify_calls.append(dafny_code)
        return VerifyOutcome(True, (), 0.0)

    monkeypatch.setattr(verifier, "verify", fake_verify)
    calls: list[tuple[FunctionRef, ...]] = []
    _patch_run_many(monkeypatch, [], calls=calls)
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 0
    assert len(verify_calls) == 1
    assert calls == []
    results = _read_results(tmp_path)
    assert len(results) == 1
    assert results[0].state is State.PROVEN
    assert results[0].reason == "certificate"
    refreshed = json.loads(sidecar.read_text(encoding="utf-8"))
    assert refreshed["pylib_sha256"] != "0" * 64
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Stale certificates" not in report
    assert capsys.readouterr().out.strip() == "1 proven"


def test_run_reproven_stale_certificate_is_not_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_repo(tmp_path)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    function = _discover(tmp_path)
    proofs = tmp_path / "proofs"
    write(from_result(_proven_result(function), Config()), proofs)
    (tmp_path / REL_PATH).write_text("def add(a: int, b: int) -> int:\n    return a - b\n", encoding="utf-8")
    stale_function = _discover(tmp_path)
    _patch_run_many(monkeypatch, [_proven_result(stale_function)])
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 0
    results = _read_results(tmp_path)
    assert len(results) == 1
    assert results[0].state is State.PROVEN
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Stale certificates" not in report
    assert "pkg/mod.py::add" in report


CALLS_SOURCE = "def g(x: int) -> int:\n    return x\n\n\ndef f(x: int) -> int:\n    return g(x)\n"
G_SPEC = "predicate spec(x: int, r: int)\n{\n  r == x\n}\n"
F_SPEC = "predicate spec(x: int, r: int)\n{\n  r == g(x)\n}\n"


def _proven_with_spec(function: FunctionRef, dafny_spec: str) -> Result:
    return Result(
        function,
        State.PROVEN,
        "verified",
        h.intent(function),
        h.spec(function),
        h.translation(function, dafny_spec=dafny_spec),
        (),
        None,
        h.proof(),
    )


def test_run_writes_callee_dependency_hashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_repo(tmp_path, CALLS_SOURCE)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    refs = {ref.qualname: ref for ref in project.discover(tmp_path, [REL_PATH])}
    assert set(refs) == {"g", "f"}
    _patch_run_many(
        monkeypatch,
        [_proven_with_spec(refs["g"], G_SPEC), _proven_with_spec(refs["f"], F_SPEC)],
    )
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 0
    sidecar = tmp_path / "proofs" / "pkg" / "mod" / "f.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["dependency_hashes"] == [{"name": "g", "sha256": hashlib.sha256(G_SPEC.encode("utf-8")).hexdigest()}]


def test_run_reruns_caller_when_callee_is_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_repo(tmp_path, CALLS_SOURCE)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    refs = {ref.qualname: ref for ref in project.discover(tmp_path, [REL_PATH])}
    proofs = tmp_path / "proofs"
    for ref in refs.values():
        write(from_result(_proven_result(ref), Config()), proofs)
    (tmp_path / REL_PATH).write_text(
        "def g(x: int) -> int:\n    return x + 1\n\n\ndef f(x: int) -> int:\n    return g(x)\n",
        encoding="utf-8",
    )
    calls: list[tuple[FunctionRef, ...]] = []

    def fake(
        functions: Sequence[FunctionRef],
        repo_root: Path,
        *,
        client: object,
        config: Config,
        callees: Mapping[FunctionRef, Result] | None = None,
    ) -> tuple[Result, ...]:
        _ = callees
        calls.append(tuple(functions))
        return tuple(Result(function, State.OPEN, "no proof found") for function in functions)

    monkeypatch.setattr(pipeline, "run_many", fake)
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 1
    assert len(calls) == 1
    assert {ref.qualname for ref in calls[0]} == {"g", "f"}


def test_run_passes_cached_callee_results_to_run_many(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_repo(tmp_path, CALLS_SOURCE)
    _patch_models(monkeypatch)
    _patch_affected_identity(monkeypatch)
    refs = {ref.qualname: ref for ref in project.discover(tmp_path, [REL_PATH])}
    assert set(refs) == {"g", "f"}
    proofs = tmp_path / "proofs"
    write(from_result(_proven_with_spec(refs["g"], G_SPEC), Config()), proofs)
    callee = refs["g"]
    caller = refs["f"]
    seen: list[Mapping[FunctionRef, Result]] = []
    _patch_run_many(
        monkeypatch,
        [Result(caller, State.OPEN, "no proof found")],
        seen_callees=seen,
    )
    assert _run(monkeypatch, tmp_path, _default_argv(tmp_path)) == 0
    assert len(seen) == 1
    assert set(seen[0]) == {callee}
    reused = seen[0][callee]
    assert reused.state is State.PROVEN
    assert reused.reason == "certificate"
    assert reused.proof is not None


# -- survey (issue #64) ----------------------------------------------------------

MIXED_SOURCES = {
    "pkg/pure.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
    "pkg/methods.py": "class Cart:\n    def total(self, xs: list[int]) -> int:\n        return sum(xs)\n",
    "pkg/untyped.py": "def double(x):\n    return x + x\n",
}

DEMO_SOURCE = (
    "import missing_pkg\n\n\ndef one(x: int) -> int:\n    return x\n\n\ndef two(x: int) -> int:\n    return x + 1\n"
)


def _write_sources(root: Path, sources: Mapping[str, str]) -> None:
    for relative, text in sources.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def _survey(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> int:
    monkeypatch.chdir(tmp_path)
    return main(argv)


def test_help_lists_survey(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "survey" in capsys.readouterr().out


def test_survey_mixed_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sources(tmp_path, MIXED_SOURCES)

    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "1 of 3 functions checkable"
    body = lines[:-1]
    assert len(body) == 3
    assert "pkg/pure.py::add: checkable" in body
    assert "pkg/methods.py::Cart.total: methods are tier 3" in body
    assert "pkg/untyped.py::double: no types" in body


def test_survey_reports_import_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sources(
        tmp_path,
        {
            "strings/demo.py": DEMO_SOURCE,
            "ok.py": "def fine(x: int) -> int:\n    return x\n",
        },
    )

    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "1 of 3 functions checkable"
    assert "ok.py::fine: checkable" in lines
    assert "strings/demo.py::one: imports strings.demo" in lines
    assert "strings/demo.py::two: imports strings.demo" in lines


def test_survey_project_calls_and_self_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sources(
        tmp_path,
        {
            "calls.py": (
                "def g(x: int) -> int:\n"
                "    return x\n"
                "\n"
                "\n"
                "def f(x: int) -> int:\n"
                "    return g(x)\n"
                "\n"
                "\n"
                "def rec(n: int) -> int:\n"
                "    return rec(n - 1) if n > 0 else 0\n"
            ),
        },
    )

    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "2 of 3 functions checkable"
    assert "calls.py::f: calls into the project come with #43" in lines
    assert "calls.py::g: checkable" in lines
    assert "calls.py::rec: checkable" in lines


def test_survey_makes_no_model_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_sources(tmp_path, MIXED_SOURCES)

    def boom(config: Config, router: object) -> Config:
        _ = (config, router)
        raise AssertionError("survey must not build a model client")

    monkeypatch.setattr(models, "build_client", boom)

    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path)]) == 0


def test_survey_empty_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == ["0 of 0 functions checkable"]


def test_survey_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sources(tmp_path, MIXED_SOURCES)

    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {"function": "pkg/methods.py::Cart.total", "reason": "methods are tier 3"},
        {"function": "pkg/pure.py::add", "reason": None},
        {"function": "pkg/untyped.py::double", "reason": "no types"},
    ]


def test_survey_json_empty_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_survey_skip_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sources(
        tmp_path,
        {
            "pkg/keep.py": "def keep(x: int) -> int:\n    return x\n",
            "tests/test_keep.py": "def skipped_a(x: int) -> int:\n    return x\n",
            "pkg/tests/deep.py": "def skipped_b(x: int) -> int:\n    return x\n",
            "docs/guide.py": "def skipped_c(x: int) -> int:\n    return x\n",
            "pkg/docs/notes.py": "def skipped_d(x: int) -> int:\n    return x\n",
            "conftest.py": "def skipped_e(x: int) -> int:\n    return x\n",
            "setup.py": "def skipped_f(x: int) -> int:\n    return x\n",
            "pkg/test_helper.py": "def skipped_g(x: int) -> int:\n    return x\n",
            "pkg/helper_test.py": "def skipped_h(x: int) -> int:\n    return x\n",
            "contest.py": "def kept_contest(x: int) -> int:\n    return x\n",
            "Tests/capital.py": "def kept_capital(x: int) -> int:\n    return x\n",
        },
    )

    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "3 of 3 functions checkable"
    assert "pkg/keep.py::keep: checkable" in lines
    assert "contest.py::kept_contest: checkable" in lines
    assert "Tests/capital.py::kept_capital: checkable" in lines
    assert all("skipped_" not in line for line in lines)


def test_survey_tool_error_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(root: Path, paths: Sequence[str]) -> tuple[FunctionRef, ...]:
        _ = (root, paths)
        raise ToolError("sandbox is down")

    monkeypatch.setattr(project, "discover", boom)

    assert _survey(monkeypatch, tmp_path, ["survey", str(tmp_path)]) == 2
    assert "sandbox is down" in capsys.readouterr().err


# -- scan (issue #65) ----------------------------------------------------------

SCAN_FIXED_SOURCE = (
    "def is_palindrome_recursive(s: str) -> bool:\n"
    '    """Check whether s reads the same forward and backward."""\n'
    "    if len(s) <= 1:\n"
    "        return True\n"
    "    if s[0] != s[-1]:\n"
    "        return False\n"
    "    return is_palindrome_recursive(s[1:-1])\n"
)

SCAN_BUGGY_SOURCE = SCAN_FIXED_SOURCE.replace("len(s) <= 1", "len(s) <= 2")


class _ScanSteps:
    """Passing step fakes in the style of #34; ``hyp_detail=None`` proves, else it counterexamples."""

    def __init__(self, hyp_detail: str | None) -> None:
        self.hyp_detail = hyp_detail
        self.review_calls = 0

    def fake_collect(self, function: FunctionRef, repo_root: Path) -> Context:
        _ = repo_root
        return Context(function, (Evidence(SourceKind.DOCSTRING, "doc", "checks the string"),))

    def fake_search_further(self, context: Context, repo_root: Path, **kwargs: Any) -> Context:
        _ = (repo_root, kwargs)
        return context

    def fake_derive(self, context: Context, **kwargs: Any) -> Intent:
        _ = kwargs
        return h.intent(context.function)

    def fake_write(self, intent: Intent, **kwargs: Any) -> Spec:
        _ = (intent, kwargs)
        return h.spec(intent.function)

    def fake_check(self, spec: Spec, function: FunctionRef, repo_root: Path, **kwargs: Any) -> Spec:
        _ = (function, repo_root, kwargs)
        if self.hyp_detail is None:
            added: tuple[CheckRecord, ...] = (
                CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 10, 10),
                CheckRecord(CheckKind.MUTANTS, True, ""),
            )
        else:
            added = (
                CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, self.hyp_detail, 10, 10),
                CheckRecord(CheckKind.MUTANTS, True, ""),
            )
        return dataclasses.replace(spec, checks=(*spec.checks, *added))

    def fake_transpile_spec(self, spec: Spec) -> str:
        _ = spec
        return "predicate spec(s: seq<char>, result: bool)"

    def fake_spec_cases(
        self, spec: Spec, function: FunctionRef, repo_root: Path, **kwargs: Any
    ) -> tuple[tuple[str, str], ...]:
        _ = (spec, function, repo_root, kwargs)
        return (('""', "true"),)

    def fake_check_spec_translation(
        self, spec: Spec, dafny_spec: str, function: FunctionRef, repo_root: Path, **kwargs: Any
    ) -> CheckRecord:
        _ = (spec, dafny_spec, function, repo_root, kwargs)
        return CheckRecord(CheckKind.DIFFERENTIAL, True, "")

    def fake_transpile(self, function: FunctionRef, **kwargs: Any) -> str:
        _ = (function, kwargs)
        return "function is_palindrome_recursive(s: seq<char>): bool"

    def fake_equivalence(
        self, function: FunctionRef, translation: Translation, repo_root: Path, **kwargs: Any
    ) -> EquivalenceReport:
        _ = (function, translation, repo_root, kwargs)
        return EquivalenceReport(records=(CheckRecord(CheckKind.DIFFERENTIAL, True, "", 10, 10),), mismatch=None)

    def fake_prove(self, translation: Translation, assumptions: Sequence[Any], **kwargs: Any) -> ProofOutcome:
        _ = (translation, assumptions, kwargs)
        return h.proof()

    def fake_search(self, function: FunctionRef, spec: Spec, repo_root: Path, **kwargs: Any) -> Counterexample | None:
        _ = (function, spec, repo_root, kwargs)
        return None

    def fake_build(self, function: FunctionRef, spec: Spec, repo_root: Path, arguments: str) -> Counterexample:
        _ = (function, spec, repo_root, arguments)
        return h.counterexample()

    def fake_review(self, context: Context, spec: Spec, **kwargs: Any) -> SecondOpinion:
        _ = (context, spec, kwargs)
        self.review_calls += 1
        return h.second_opinion()


def _install_scan_steps(monkeypatch: pytest.MonkeyPatch, hyp_detail: str | None) -> _ScanSteps:
    steps = _ScanSteps(hyp_detail)
    monkeypatch.setattr(pipeline, "collect", steps.fake_collect)
    monkeypatch.setattr(pipeline, "search_further", steps.fake_search_further)
    monkeypatch.setattr(pipeline, "derive", steps.fake_derive)
    monkeypatch.setattr(pipeline, "write_spec", steps.fake_write)
    monkeypatch.setattr(pipeline, "check_spec", steps.fake_check)
    monkeypatch.setattr(pipeline, "transpile_spec", steps.fake_transpile_spec)
    monkeypatch.setattr(pipeline, "spec_cases", steps.fake_spec_cases)
    monkeypatch.setattr(pipeline, "check_spec_translation", steps.fake_check_spec_translation)
    monkeypatch.setattr(pipeline, "transpile_function", steps.fake_transpile)
    monkeypatch.setattr(pipeline, "check_equivalence", steps.fake_equivalence)
    monkeypatch.setattr(pipeline, "prove", steps.fake_prove)
    monkeypatch.setattr(pipeline, "search_counterexample", steps.fake_search)
    monkeypatch.setattr(pipeline, "build_counterexample", steps.fake_build)
    monkeypatch.setattr(pipeline, "review", steps.fake_review)
    return steps


def _scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> int:
    monkeypatch.chdir(tmp_path)
    return main(argv)


def _scan_entries(out_path: Path) -> list[Any]:
    return [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_scan_help_lists_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["scan", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in ("--out", "--resume", "--contributor"):
        assert option in out


def test_scan_without_repo_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["scan"])
    assert exc.value.code == 2


def test_scan_fixed_palindrome_is_proven(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sources(tmp_path, {"strings/palindrome.py": SCAN_FIXED_SOURCE})
    _patch_models(monkeypatch)
    steps = _install_scan_steps(monkeypatch, hyp_detail=None)

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path)]) == 0
    assert steps.review_calls == 0

    out_path = tmp_path / "scan.jsonl"
    entries = _scan_entries(out_path)
    assert len(entries) == 1
    assert entries[0]["project"] == tmp_path.name
    result = serialize.from_data(Result, entries[0]["result"])
    assert result.state is State.PROVEN
    out = capsys.readouterr().out
    assert "strings/palindrome.py::is_palindrome_recursive" in out
    assert "proven" in out


def test_scan_buggy_palindrome_is_counterexample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sources(tmp_path, {"strings/palindrome.py": SCAN_BUGGY_SOURCE})
    _patch_models(monkeypatch)
    steps = _install_scan_steps(monkeypatch, hyp_detail='"ab"')

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path)]) == 1
    assert steps.review_calls == 1

    entries = _scan_entries(tmp_path / "scan.jsonl")
    assert len(entries) == 1
    result = serialize.from_data(Result, entries[0]["result"])
    assert result.state is State.COUNTEREXAMPLE
    assert "wrong value" in capsys.readouterr().out


def test_scan_counterexample_shows_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sources(tmp_path, {"strings/palindrome.py": SCAN_BUGGY_SOURCE})
    _patch_models(monkeypatch)
    _install_scan_steps(monkeypatch, hyp_detail="'ab' raises ValueError")

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path)]) == 1

    entries = _scan_entries(tmp_path / "scan.jsonl")
    assert len(entries) == 1
    result = serialize.from_data(Result, entries[0]["result"])
    assert result.state is State.COUNTEREXAMPLE
    assert "raises ValueError" in capsys.readouterr().out


def test_scan_runs_pipeline_with_scan_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sources(tmp_path, {"pkg/mod.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
    _patch_models(monkeypatch)
    seen: list[Config] = []
    graph_calls: list[int] = []
    import_calls: list[int] = []
    real_graph = project.call_graph
    real_imports = pipeline.module_import_errors

    def counting_graph(root: Path, functions: Sequence[FunctionRef]) -> Mapping[FunctionRef, frozenset[FunctionRef]]:
        graph_calls.append(1)
        return real_graph(root, functions)

    def counting_imports(functions: Sequence[FunctionRef], root: Path) -> dict[str, str]:
        import_calls.append(1)
        return real_imports(functions, root)

    def fake_run(function: FunctionRef, repo_root: Path, *, client: object, config: Config) -> Result:
        _ = (repo_root, client)
        seen.append(config)
        return Result(function, State.OPEN, "no proof found")

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(project, "call_graph", counting_graph)
    monkeypatch.setattr(pipeline, "module_import_errors", counting_imports)

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path)]) == 0
    assert len(seen) == 1
    assert seen[0].pipeline.scan is True
    assert graph_calls == [1]
    assert import_calls == [1]
    assert "pkg/mod.py::add" in capsys.readouterr().out


def test_scan_tool_error_becomes_open_and_scan_goes_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sources(
        tmp_path,
        {
            "pkg/a.py": "def a(x: int) -> int:\n    return x\n",
            "pkg/b.py": "def b(x: int) -> int:\n    return x\n",
            "pkg/c.py": "def c(x: int) -> int:\n    return x\n",
        },
    )
    _patch_models(monkeypatch)

    def fake_run(function: FunctionRef, repo_root: Path, *, client: object, config: Config) -> Result:
        _ = (repo_root, client, config)
        if function.qualname == "b":
            raise ToolError("model API down")
        return Result(function, State.OPEN, "no proof found")

    monkeypatch.setattr(pipeline, "run", fake_run)

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path)]) == 0

    entries = _scan_entries(tmp_path / "scan.jsonl")
    assert len(entries) == 3
    results = [serialize.from_data(Result, entry["result"]) for entry in entries]
    assert [result.function.qualname for result in results] == ["a", "b", "c"]
    assert results[1].state is State.OPEN
    assert results[1].reason == "tool error: model API down"
    assert results[0].state is State.OPEN
    assert results[2].state is State.OPEN
    assert "tool error: model API down" in capsys.readouterr().out


def test_scan_resume_skips_recorded_functions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_sources(tmp_path, {"pkg/mod.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
    _patch_models(monkeypatch)

    def fake_run(function: FunctionRef, repo_root: Path, *, client: object, config: Config) -> Result:
        _ = (repo_root, client, config)
        return Result(function, State.OPEN, "no proof found")

    monkeypatch.setattr(pipeline, "run", fake_run)
    out = tmp_path / "scan.jsonl"
    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path), "--out", str(out)]) == 0
    before = out.read_bytes()

    def boom(function: FunctionRef, repo_root: Path, **kwargs: Any) -> Result:
        _ = (function, repo_root, kwargs)
        raise AssertionError("resumed functions must not run again")

    monkeypatch.setattr(pipeline, "run", boom)
    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path), "--out", str(out), "--resume"]) == 0
    assert out.read_bytes() == before


def test_scan_resume_without_out_starts_new_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_sources(tmp_path, {"pkg/mod.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
    _patch_models(monkeypatch)
    calls: list[str] = []

    def fake_run(function: FunctionRef, repo_root: Path, *, client: object, config: Config) -> Result:
        _ = (repo_root, client, config)
        calls.append(function.qualname)
        return Result(function, State.OPEN, "no proof found")

    monkeypatch.setattr(pipeline, "run", fake_run)
    out = tmp_path / "scan.jsonl"
    assert not out.exists()

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path), "--out", str(out), "--resume"]) == 0
    assert calls == ["add"]
    assert len(_scan_entries(out)) == 1


def test_scan_out_lines_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "pkg/ok.py": "def ok(x: int) -> int:\n    return x\n",
            "pkg/untyped.py": "def vague(x):\n    return x\n",
        },
    )
    _patch_models(monkeypatch)

    def fake_run(function: FunctionRef, repo_root: Path, *, client: object, config: Config) -> Result:
        _ = (repo_root, client, config)
        if function.qualname == "ok":
            return Result(
                function,
                State.PROVEN,
                "",
                h.intent(function),
                h.spec(function),
                h.translation(function),
                (),
                None,
                h.proof(),
            )
        return Result(function, State.OPEN, "no proof found")

    monkeypatch.setattr(pipeline, "run", fake_run)
    out = tmp_path / "scan.jsonl"

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path), "--out", str(out)]) == 0

    raw = out.read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert len(lines) == 2
    assert raw.count("\n") == len(lines)
    for line in lines:
        entry = json.loads(line)
        assert set(entry) == {"project", "result"}
        assert entry["project"] == tmp_path.name
        result = serialize.from_data(Result, entry["result"])
        assert result.state in (State.PROVEN, State.NOT_CHECKABLE)


def test_run_contributor_is_usage_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["run", "--contributor"])
    assert exc.value.code == 2


def test_scan_contributor_rewrites_spark_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_sources(tmp_path, {"pkg/mod.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
    _patch_models(monkeypatch)
    seen: list[Config] = []

    def fake_run(function: FunctionRef, repo_root: Path, *, client: object, config: Config) -> Result:
        _ = (repo_root, client)
        seen.append(config)
        return Result(function, State.OPEN, "no proof found")

    monkeypatch.setattr(pipeline, "run", fake_run)

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path), "--contributor"]) == 0
    assert len(seen) == 1
    rewritten = seen[0].models
    assert rewritten.intent == "muse-spark-1.3-contributor"
    assert rewritten.spec == "muse-spark-1.3-contributor"
    assert rewritten.mutants == "muse-spark-1.3-contributor"
    assert rewritten.translation == "muse-spark-1.3-contributor"
    assert rewritten.edge_cases == "muse-spark-1.3-contributor"
    assert rewritten.counterexample == "muse-spark-1.3-contributor"
    assert rewritten.second_opinion == "muse-spark-1.3-contributor"
    assert rewritten.context_search == "muse-spark-1.3-contributor"
    assert [stage.model for stage in seen[0].proof.prover_stages] == [
        "glimmer",
        "muse-spark-1.3-contributor",
        "muse-wonder-1.0",
    ]


def test_scan_empty_repo_exits_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_models(monkeypatch)
    out = tmp_path / "scan.jsonl"

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == ""


def test_scan_reports_precheck_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sources(tmp_path, {"pkg/untyped.py": "def vague(x):\n    return x\n"})
    _patch_models(monkeypatch)

    def boom(function: FunctionRef, repo_root: Path, **kwargs: Any) -> Result:
        _ = (function, repo_root, kwargs)
        raise AssertionError("not-checkable functions must not run")

    monkeypatch.setattr(pipeline, "run", boom)

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path)]) == 0
    entries = _scan_entries(tmp_path / "scan.jsonl")
    assert len(entries) == 1
    result = serialize.from_data(Result, entries[0]["result"])
    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "no types"
    assert "no types" in capsys.readouterr().out


def test_scan_runs_callees_before_callers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {"pkg/calls.py": "def g(x: int) -> int:\n    return x\n\n\ndef f(x: int) -> int:\n    return g(x)\n"},
    )
    _patch_models(monkeypatch)
    calls: list[str] = []

    def fake_run(function: FunctionRef, repo_root: Path, *, client: object, config: Config) -> Result:
        _ = (repo_root, client, config)
        calls.append(function.qualname)
        return Result(function, State.OPEN, "no proof found")

    monkeypatch.setattr(pipeline, "run", fake_run)

    assert _scan(monkeypatch, tmp_path, ["scan", str(tmp_path)]) == 0
    assert calls == ["g"]
    entries = _scan_entries(tmp_path / "scan.jsonl")
    results = [serialize.from_data(Result, entry["result"]) for entry in entries]
    assert [result.function.qualname for result in results] == ["g", "f"]
    assert results[1].state is State.NOT_CHECKABLE
    assert results[1].reason == "calls into the project come with #43"
