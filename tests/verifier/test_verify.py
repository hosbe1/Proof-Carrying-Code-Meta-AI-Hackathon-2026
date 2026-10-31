"""Tests for ``pcc.verifier.dafny.verify``."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from pcc.config import Config
from pcc.core import DafnyError, ToolError
from pcc.verifier import dafny as dafny_module
from pcc.verifier.dafny import verify

RESOURCE_LIMIT = Config().dafny.resource_limit

VERIFIED = "method Add(x: int, y: int) returns (r: int)\n  ensures r == x + y\n{\n  r := x + y;\n}\n"

WRONG_ENSURES = "method Add(x: int, y: int) returns (r: int)\n  ensures r > x + y\n{\n  r := x + y;\n}\n"

AXIOM_CHEAT = "method M() ensures false\n{\n  assume {:axiom} false;\n}\n"


@pytest.mark.dafny
def test_verify_returns_verified_for_correct_method() -> None:
    outcome = verify(VERIFIED, resource_limit=RESOURCE_LIMIT)
    assert outcome.verified is True
    assert outcome.errors == ()
    assert outcome.seconds >= 0.0


@pytest.mark.dafny
def test_verify_reports_unprovable_ensures_with_related_location() -> None:
    outcome = verify(WRONG_ENSURES, resource_limit=RESOURCE_LIMIT)
    assert outcome.verified is False
    assert len(outcome.errors) == 1
    assert "ensures r > x + y" in outcome.errors[0].related


@pytest.mark.dafny
def test_verify_rejects_axiom_cheat_despite_exit_zero() -> None:
    outcome = verify(AXIOM_CHEAT, resource_limit=RESOURCE_LIMIT)
    assert outcome.verified is False
    assert outcome.errors != ()
    assert all(error.message.startswith("anti-cheat: ") for error in outcome.errors)
    assert any("assume" in error.message for error in outcome.errors)


def test_verify_runs_dafny_verify_with_resource_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Sequence[str], float | None, Path]] = []
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(args: Sequence[str], *, timeout_seconds: float | None, cwd: Path) -> subprocess.CompletedProcess[str]:
        seen.append((tuple(args), timeout_seconds, cwd))
        assert (cwd / "main.dfy").read_text(encoding="utf-8") == VERIFIED
        assert (cwd / "PyLib.dfy").is_file()
        return subprocess.CompletedProcess(args=["dafny", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    outcome = verify(VERIFIED, resource_limit=123, timeout_seconds=None)
    assert outcome.verified is True
    assert outcome.errors == ()
    assert [list(args) for args, _, _ in seen] == [
        ["verify", "--resource-limit", "123", "--verification-time-limit", "0", "main.dfy", "PyLib.dfy"]
    ]
    assert outcome.seconds >= 0.0


def test_verify_timeout_reports_zero_position(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(args: Sequence[str], *, timeout_seconds: float | None, cwd: Path) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["dafny", *args], timeout=timeout_seconds or 0.0)

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    outcome = verify(VERIFIED, resource_limit=RESOURCE_LIMIT, timeout_seconds=0.001)
    assert outcome.verified is False
    assert outcome.errors == (DafnyError(0, 0, "timeout after 0.001 s"),)
    assert outcome.seconds >= 0.0


def test_verify_exit_code_without_errors_falls_back_to_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(args: Sequence[str], *, timeout_seconds: float | None, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["dafny", *args], returncode=4, stdout="", stderr="boom")

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    outcome = verify(VERIFIED, resource_limit=RESOURCE_LIMIT)
    assert outcome.verified is False
    assert outcome.errors == (DafnyError(0, 0, "boom"),)


def test_verify_maps_anti_cheat_violations_to_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(args: Sequence[str], *, timeout_seconds: float | None, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["dafny", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    code = "method M() {\n  assume false;\n}\n"
    outcome = verify(code, resource_limit=RESOURCE_LIMIT)
    assert outcome.verified is False
    assert outcome.errors == (DafnyError(2, 0, "anti-cheat: assume"),)


def test_verify_without_dafny_on_path_raises_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ToolError, match="dafny"):
        verify(VERIFIED, resource_limit=RESOURCE_LIMIT)


def test_verify_with_wrong_version_raises_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "dafny"
    fake.write_text("#!/bin/sh\necho '9.9.9'\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ToolError, match=r"9\.9\.9"):
        verify(VERIFIED, resource_limit=RESOURCE_LIMIT)
