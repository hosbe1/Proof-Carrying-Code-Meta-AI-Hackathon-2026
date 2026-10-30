"""Tests for ``pcc.project.infer_types``: parameter and return types from pyright.

The ticket examples run against the real ``pyright`` CLI and skip when it is not
installed. A fake ``pyright`` executable on ``PATH`` covers the ``--outputjson``
parsing, so the probe and the signature parser are tested everywhere.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

import pytest

from pcc.core import FunctionRef
from pcc.project import infer_types
from pcc.project.discover import discover

_NEEDS_PYRIGHT = pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright CLI not on PATH")

_FAKE_PYRIGHT_SCRIPT = """\
#!__EXECUTABLE__
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    mode = os.environ.get("FAKE_PYRIGHT_MODE", "reveal")
    if mode == "garbage":
        print("this is not JSON")
        return
    target = sys.argv[-1]
    diagnostics = [
        {
            "file": target,
            "severity": "information",
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
            "message": os.environ["FAKE_PYRIGHT_MESSAGE"],
        },
        {
            "file": target,
            "severity": "error",
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
            "message": "unrelated noise the parser must ignore",
        },
    ]
    print(json.dumps({"version": "fake", "generalDiagnostics": diagnostics, "summary": {}}))
    if mode == "error-exit":
        raise SystemExit(1)


main()
"""


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ref(repo: Path, rel: str, qualname: str) -> FunctionRef:
    refs = {ref.qualname: ref for ref in discover(repo, [rel])}
    return refs[qualname]


def _install_fake_pyright(bin_dir: Path) -> None:
    """An executable ``pyright`` that reports ``FAKE_PYRIGHT_MESSAGE`` for the analysed file."""
    script = bin_dir / "pyright"
    script.write_text(_FAKE_PYRIGHT_SCRIPT.replace("__EXECUTABLE__", sys.executable), encoding="utf-8")
    script.chmod(0o755)


def _with_fake_pyright(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, message: str, mode: str = "reveal") -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_fake_pyright(bin_dir)
    monkeypatch.setenv("FAKE_PYRIGHT_MESSAGE", message)
    monkeypatch.setenv("FAKE_PYRIGHT_MODE", mode)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))


@_NEEDS_PYRIGHT
def test_default_value_types_are_inferred(tmp_path: Path) -> None:
    """Example: ``def f(x=0): return x + 1`` gives ``{"x": "int", "return": "int"}``."""
    _write(tmp_path, "mod.py", "def f(x=0):\n    return x + 1\n")
    assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) == {"x": "int", "return": "int"}


@_NEEDS_PYRIGHT
def test_uninferrable_types_return_none(tmp_path: Path) -> None:
    """Example: ``def f(x): return x`` gives ``None`` because every type is ``Unknown``."""
    _write(tmp_path, "mod.py", "def f(x):\n    return x\n")
    assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) is None


def test_fully_annotated_returns_without_pyright(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Example: a fully annotated function returns its annotations without running pyright."""

    def _fail_which(*args: object, **kwargs: object) -> str | None:
        raise AssertionError("pyright must not run for a fully annotated function")

    monkeypatch.setattr(shutil, "which", _fail_which)
    _write(
        tmp_path,
        "mod.py",
        'def f(x: int, *args: int, y: str = "a", **kwargs: str) -> bool:\n    return True\n',
    )
    assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) == {
        "x": "int",
        "args": "int",
        "y": "str",
        "kwargs": "str",
        "return": "bool",
    }


def test_missing_pyright_warns_and_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Edge case: without pyright on ``PATH`` the result is ``None`` with a warning."""
    monkeypatch.setenv("PATH", str(tmp_path))
    _write(tmp_path, "mod.py", "def f(x=0):\n    return x + 1\n")
    with caplog.at_level(logging.WARNING, logger="pcc.project.types"):
        assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) is None
    assert any(record.levelno == logging.WARNING and "pyright" in record.message for record in caplog.records)


def test_reveal_output_is_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe answer ``(x: int = 0) -> int`` parses to parameter and return types."""
    _with_fake_pyright(monkeypatch, tmp_path, 'Type of "f" is "(x: int = 0) -> int"')
    _write(tmp_path, "mod.py", "def f(x=0):\n    return x + 1\n")
    assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) == {"x": "int", "return": "int"}


def test_partial_annotations_are_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Annotated parameters keep their types while the rest is inferred."""
    _with_fake_pyright(monkeypatch, tmp_path, 'Type of "f" is "(x: int, y: int = 0) -> int"')
    _write(tmp_path, "mod.py", "def f(x: int, y=0):\n    return x + y\n")
    assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) == {
        "x": "int",
        "y": "int",
        "return": "int",
    }


def test_unknown_type_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any ``Unknown`` in the revealed signature gives ``None``."""
    _with_fake_pyright(monkeypatch, tmp_path, 'Type of "f" is "(x: Unknown) -> Unknown"')
    _write(tmp_path, "mod.py", "def f(x):\n    return x\n")
    assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) is None


def test_garbage_output_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Output that is no JSON gives ``None`` with a warning, never an exception."""
    _with_fake_pyright(monkeypatch, tmp_path, "", mode="garbage")
    _write(tmp_path, "mod.py", "def f(x=0):\n    return x + 1\n")
    with caplog.at_level(logging.WARNING, logger="pcc.project.types"):
        assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) is None
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_error_exit_still_parses_reveal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero pyright exit still yields types when the reveal diagnostic is there."""
    _with_fake_pyright(monkeypatch, tmp_path, 'Type of "f" is "(x: int = 0) -> int"', mode="error-exit")
    _write(tmp_path, "mod.py", "def f(x=0):\n    return x + 1\n")
    assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) == {"x": "int", "return": "int"}


def test_lambda_default_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``lambda`` default hides commas from the parser, so the result is ``None``."""
    _with_fake_pyright(monkeypatch, tmp_path, 'Type of "f" is "(cb: Callable[[int], int] = lambda a, b: a) -> int"')
    _write(tmp_path, "mod.py", "def f(cb=lambda a, b: a):\n    return cb(1)\n")
    assert infer_types(tmp_path, _ref(tmp_path, "mod.py", "f")) is None
