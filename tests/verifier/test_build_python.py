"""Tests for ``pcc.verifier.dafny.build_python`` and ``python_name``."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from pcc.core import NotCheckable
from pcc.sandbox import run_python
from pcc.verifier import dafny as dafny_module
from pcc.verifier.dafny import build_python, python_name

PALINDROME = """module Palindrome {
  function is_palindrome_recursive(s: string): bool {
    if |s| <= 1 then true else s[0] == s[|s|-1] && is_palindrome_recursive(s[1..|s|-1])
  }
}
"""

TYPE_ERROR = 'method M() {\n  var x: int := "hello";\n}\n'

UNPROVABLE = "method Add(x: int, y: int) returns (r: int)\n  ensures r > x + y\n{\n  r := x + y;\n}\n"

_GENERATED_MODULES = ("Palindrome", "ProbeMod", "My__Module", "_dafny", "System_", "module_")


@pytest.fixture
def _clean_generated() -> Iterator[None]:
    try:
        yield
    finally:
        for name in _GENERATED_MODULES:
            sys.modules.pop(name, None)


def _import_module(module_dir: Path, name: str) -> ModuleType:
    sys.path.insert(0, str(module_dir))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(module_dir))


@pytest.mark.dafny
def test_build_python_palindrome_returns_true_in_sandbox(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    module_dir = build_python(PALINDROME, out_dir)
    assert module_dir == out_dir / "m-py"
    assert module_dir.is_dir()
    script = (
        f"import sys; sys.path.insert(0, {str(module_dir)!r})\n"
        "import Palindrome\n"
        "result = Palindrome.default__.is__palindrome__recursive('aba')\n"
        "assert result is True, result\n"
    )
    outcome = run_python(script, cwd=tmp_path)
    assert outcome.ok, outcome.stderr + outcome.stdout


@pytest.mark.dafny
def test_build_python_type_error_raises_not_checkable(tmp_path: Path) -> None:
    with pytest.raises(NotCheckable, match="translation does not compile"):
        build_python(TYPE_ERROR, tmp_path / "out")


@pytest.mark.dafny
def test_build_python_type_error_reports_first_error(tmp_path: Path) -> None:
    with pytest.raises(NotCheckable) as excinfo:
        build_python(TYPE_ERROR, tmp_path / "out")
    assert "translation does not compile" in str(excinfo.value)
    assert "not assignable" in str(excinfo.value)


@pytest.mark.dafny
def test_build_python_unverifiable_code_still_compiles(tmp_path: Path) -> None:
    module_dir = build_python(UNPROVABLE, tmp_path / "out")
    assert module_dir.is_dir()
    assert next(module_dir.glob("*.py")) is not None


@pytest.mark.dafny
def test_build_python_clears_nonempty_out_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")
    nested = out_dir / "nested"
    nested.mkdir()
    (nested / "old.py").write_text("x = 1", encoding="utf-8")
    module_dir = build_python(PALINDROME, out_dir)
    assert module_dir.is_dir()
    assert not (out_dir / "stale.txt").exists()
    assert not nested.exists()


def test_build_python_runs_build_without_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[tuple[tuple[str, ...], Path]] = []
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(
        args: list[str] | tuple[str, ...],
        *,
        timeout_seconds: float | None,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        seen.append((tuple(args), cwd))
        assert (cwd / "main.dfy").is_file()
        assert (cwd / "PyLib.dfy").is_file()
        return subprocess.CompletedProcess(args=["dafny", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")
    module_dir = build_python(PALINDROME, out_dir)
    assert module_dir == out_dir / "m-py"
    assert not (out_dir / "stale.txt").exists()
    assert len(seen) == 1
    args, _ = seen[0]
    assert args[:3] == ("build", "--target:py", "--no-verify")
    assert "--output" in args
    output_index = args.index("--output")
    assert args[output_index + 1].endswith("/m")
    assert args[-2:] == ("main.dfy", "PyLib.dfy")


def test_build_python_build_failure_uses_first_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(
        args: list[str] | tuple[str, ...],
        *,
        timeout_seconds: float | None,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        stdout = "main.dfy(2,13): Error: first problem\nmain.dfy(3,1): Error: second problem\n"
        return subprocess.CompletedProcess(args=["dafny", *args], returncode=2, stdout=stdout, stderr="")

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    with pytest.raises(NotCheckable) as excinfo:
        build_python(TYPE_ERROR, tmp_path / "out")
    message = str(excinfo.value)
    assert message.startswith("translation does not compile: ")
    assert "first problem" in message
    assert "second problem" not in message


def test_build_python_unparseable_output_falls_back_to_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dafny_module, "_require_dafny", lambda: None)

    def fake_run(
        args: list[str] | tuple[str, ...],
        *,
        timeout_seconds: float | None,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["dafny", *args], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(dafny_module, "_run_dafny", fake_run)
    with pytest.raises(NotCheckable, match="translation does not compile: boom"):
        build_python(TYPE_ERROR, tmp_path / "out")


def test_build_python_without_dafny_on_path_raises_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from pcc.core import ToolError

    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ToolError, match="dafny"):
        build_python(PALINDROME, tmp_path / "out")


def test_python_name_doubles_underscores() -> None:
    assert python_name("py_floordiv") == "py__floordiv"
    assert python_name("is_palindrome_recursive") == "is__palindrome__recursive"
    assert python_name("foo") == "foo"
    assert python_name("foo_bar_baz") == "foo__bar__baz"
    assert python_name("a__b") == "a____b"


def test_python_name_escapes_reserved_words() -> None:
    assert python_name("global") == "global_"
    assert python_name("lambda") == "lambda_"
    assert python_name("len") == "len_"
    assert python_name("list") == "list_"


@pytest.mark.dafny
def test_python_name_matches_build_output(tmp_path: Path, _clean_generated: None) -> None:
    assert python_name("py_floordiv") == "py__floordiv"
    assert python_name("is_palindrome_recursive") == "is__palindrome__recursive"
    assert python_name("My_Module") == "My__Module"
    code = (
        "module ProbeMod {\n"
        "  function py_floordiv(x: int): int { x }\n"
        "  function is_palindrome_recursive(x: int): int { x }\n"
        "}\n"
        "module My_Module {\n"
        "  function hello(x: int): int { x }\n"
        "}\n"
    )
    module_dir = build_python(code, tmp_path / "out")
    assert (module_dir / (python_name("My_Module") + ".py")).is_file()
    probe = _import_module(module_dir, "ProbeMod")
    assert hasattr(probe.default__, python_name("py_floordiv"))
    assert hasattr(probe.default__, python_name("is_palindrome_recursive"))
