"""Tests for ``pcc-check run`` (issue #40).

Every example and edge case of the ticket runs against a temporary repository with the pipeline
monkeypatched; only ``discover``, the certificate store and the report run for real.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import pytest

from pcc import models, pipeline, project, verifier
from pcc.certificate import from_result, write
from pcc.cli import main
from pcc.config import Config
from pcc.core import FunctionRef, Result, State, ToolError, serialize
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
) -> None:
    def fake(
        functions: Sequence[FunctionRef],
        repo_root: Path,
        *,
        client: object,
        config: Config,
    ) -> tuple[Result, ...]:
        if calls is not None:
            calls.append(tuple(functions))
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
