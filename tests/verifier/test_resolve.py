"""Tests for ``pcc.verifier.dafny.resolve``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcc.core import DafnyError, ToolError
from pcc.verifier.dafny import resolve

UNDECLARED = "method M() {\n  var x := UndefinedName;\n}\n"

USING_PYLIB = (
    'include "PyLib.dfy"\n'
    "module ResolveGoodFixture {\n"
    "  import opened PyLib\n"
    "  function f(a: int, b: int): int\n"
    "    requires b != 0\n"
    "  {\n"
    "    py_floordiv(a, b)\n"
    "  }\n"
    "}\n"
)


@pytest.mark.dafny
def test_resolve_returns_error_for_undeclared_name() -> None:
    errors = resolve(UNDECLARED)
    assert len(errors) == 1
    error = errors[0]
    assert (error.line, error.column) == (2, 11)
    assert error.message == "unresolved identifier: UndefinedName"


@pytest.mark.dafny
def test_resolve_returns_empty_for_valid_module_using_pylib() -> None:
    assert resolve(USING_PYLIB) == ()


@pytest.mark.dafny
def test_resolve_timeout_reports_zero_position() -> None:
    assert resolve("method M() {\n}\n", timeout_seconds=0.001) == (DafnyError(0, 0, "timeout after 0.001 s"),)


def test_resolve_without_dafny_on_path_raises_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ToolError, match="dafny"):
        resolve(UNDECLARED)


def test_resolve_with_wrong_version_raises_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "dafny"
    fake.write_text("#!/bin/sh\necho '9.9.9'\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ToolError, match=r"9\.9\.9"):
        resolve(UNDECLARED)
