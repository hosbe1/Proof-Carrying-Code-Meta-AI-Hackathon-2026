"""Tests for verifying a caller together with proven callee modules."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from pcc.config import Config
from pcc.core import FunctionRef, source_hash
from pcc.verifier import dafny as dafny_module
from pcc.verifier.dafny import verify
from pcc.verifier.layout import assemble, module_name

RESOURCE_LIMIT = Config().dafny.resource_limit

CALLEE = "module Callee {\n  function Double(x: int): int\n    ensures Double(x) == 2 * x\n  {\n    2 * x\n  }\n}\n"

BROKEN_CALLEE = (
    "module Callee {\n  function Double(x: int): int\n    ensures Double(x) == 2 * x + 1\n  {\n    2 * x\n  }\n}\n"
)

UNRESOLVABLE_EXTRA = "module Broken {\n  method M() {\n    var x := UndefinedName;\n  }\n}\n"

CALLER_METHOD = "method Caller() returns (r: int)\n  ensures r == 42\n{\n  r := Callee.Double(21);\n}\n"

TRIVIAL_METHOD = "method Add(x: int, y: int) returns (r: int)\n  ensures r == x + y\n{\n  r := x + y;\n}\n"


def make_ref(path: str = "pkg/mod.py", qualname: str = "caller") -> FunctionRef:
    source = "def caller():\n    return 0\n"
    return FunctionRef(path=path, qualname=qualname, lineno=1, source=source, source_hash=source_hash(source))


def test_assemble_emits_import_lines_after_pylib_import() -> None:
    function = make_ref()
    assembled = assemble(function, "", CALLER_METHOD, imports=["Callee"])
    lines = assembled.split("\n")
    assert lines[0] == f"module {module_name(function)} {{"
    assert lines[1] == "  import opened PyLib"
    assert lines[2] == "  import Callee"
    assert "Callee.Double(21)" in assembled


def test_assemble_keeps_import_order_and_drops_duplicates() -> None:
    function = make_ref()
    assembled = assemble(function, "", CALLER_METHOD, imports=["Callee", "Other", "Callee"])
    import_lines = [line for line in assembled.split("\n") if line.strip().startswith("import ")]
    assert import_lines == ["  import opened PyLib", "  import Callee", "  import Other"]


def test_verify_writes_extra_modules_as_dep_files_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Sequence[str], Path]] = []
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(args: Sequence[str], *, timeout_seconds: float | None, cwd: Path) -> subprocess.CompletedProcess[str]:
        seen.append((tuple(args), cwd))
        assert (cwd / "main.dfy").read_text(encoding="utf-8") == TRIVIAL_METHOD
        assert (cwd / "PyLib.dfy").is_file()
        assert (cwd / "dep0.dfy").read_text(encoding="utf-8") == "module A {}\n"
        assert (cwd / "dep1.dfy").read_text(encoding="utf-8") == "module B {}\n"
        return subprocess.CompletedProcess(args=["dafny", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    outcome = verify(TRIVIAL_METHOD, resource_limit=123, extra_modules=("module A {}\n", "module B {}\n"))
    assert outcome.verified is True
    assert outcome.errors == ()
    assert [list(args) for args, _ in seen] == [
        [
            "verify",
            "--resource-limit",
            "123",
            "--verification-time-limit",
            "0",
            "main.dfy",
            "PyLib.dfy",
            "dep0.dfy",
            "dep1.dfy",
        ]
    ]


def test_verify_ignores_cheat_markers_in_extra_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(args: Sequence[str], *, timeout_seconds: float | None, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["dafny", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    outcome = verify(TRIVIAL_METHOD, resource_limit=RESOURCE_LIMIT, extra_modules=("module A {\n  assume false;\n}\n",))
    assert outcome.verified is True
    assert outcome.errors == ()


@pytest.mark.dafny
def test_caller_with_verified_callee_verifies() -> None:
    assembled = assemble(make_ref(), "", CALLER_METHOD, imports=["Callee"])
    outcome = verify(assembled, resource_limit=RESOURCE_LIMIT, extra_modules=(CALLEE,))
    assert outcome.verified is True
    assert outcome.errors == ()


@pytest.mark.dafny
def test_broken_callee_makes_caller_unverified() -> None:
    assembled = assemble(make_ref(), "", CALLER_METHOD, imports=["Callee"])
    outcome = verify(assembled, resource_limit=RESOURCE_LIMIT, extra_modules=(BROKEN_CALLEE,))
    assert outcome.verified is False
    assert outcome.errors != ()


@pytest.mark.dafny
def test_caller_without_extra_module_has_resolve_error() -> None:
    assembled = assemble(make_ref(), "", CALLER_METHOD, imports=["Callee"])
    outcome = verify(assembled, resource_limit=RESOURCE_LIMIT)
    assert outcome.verified is False
    assert outcome.errors != ()
    assert any("Callee" in error.message for error in outcome.errors)


@pytest.mark.dafny
def test_unresolvable_extra_module_is_not_verified() -> None:
    outcome = verify(TRIVIAL_METHOD, resource_limit=RESOURCE_LIMIT, extra_modules=(UNRESOLVABLE_EXTRA,))
    assert outcome.verified is False
    assert outcome.errors != ()
    assert any("UndefinedName" in error.message for error in outcome.errors)
