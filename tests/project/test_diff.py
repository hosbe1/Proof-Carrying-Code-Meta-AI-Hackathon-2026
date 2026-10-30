"""Tests for ``pcc.project.diff.changed``: which functions a pull request changed.

Every test builds a temporary git repository, so the diff mapping is checked end to end.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pcc.core import ToolError
from pcc.project import changed


def _git(repo: Path, *args: str) -> str:
    """Run ``git <args>`` in ``repo``; fail the test when git itself fails."""
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        cwd=str(repo),
        check=False,
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr}"
    return proc.stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit_all(repo: Path, message: str) -> None:
    # ``--no-verify`` skips local hooks of the machine the test runs on; the
    # fixture repositories carry no hooks of their own.
    _git(repo, "add", "-A")
    _git(repo, "commit", "--no-verify", "-m", message)


def _init_repo(path: Path) -> str:
    """A git repository with one commit; returns the base commit sha."""
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _write(path, "mod.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    _commit_all(path, "initial")
    return _git(path, "rev-parse", "HEAD")


def _qualnames(repo: Path, base: str) -> list[tuple[str, str]]:
    return [(ref.path, ref.qualname) for ref in changed(repo, base)]


def test_body_change_returns_only_that_function(tmp_path: Path) -> None:
    """Example: changing the body of ``g`` reports ``g`` but not ``f``."""
    base = _init_repo(tmp_path)
    _write(tmp_path, "mod.py", "def f():\n    return 1\n\n\ndef g():\n    return 3\n")
    _commit_all(tmp_path, "change g")
    assert _qualnames(tmp_path, base) == [("mod.py", "g")]


def test_change_outside_any_function_returns_empty(tmp_path: Path) -> None:
    """A hunk in a module constant touches no function range."""
    base = _init_repo(tmp_path)
    _write(
        tmp_path,
        "mod.py",
        "LIMIT = 99\n\n\ndef f():\n    return 1\n\n\ndef g():\n    return 2\n",
    )
    _commit_all(tmp_path, "change constant")
    assert changed(tmp_path, base) == ()


def test_decorator_change_marks_the_function(tmp_path: Path) -> None:
    """The range starts at the first decorator line, so a decorator hunk counts."""
    base = _init_repo(tmp_path)
    _write(
        tmp_path,
        "mod.py",
        "from functools import cache\n\n\n@cache\ndef f():\n    return 1\n\n\ndef g():\n    return 2\n",
    )
    _commit_all(tmp_path, "decorate f")
    assert _qualnames(tmp_path, base) == [("mod.py", "f")]


def test_new_file_reports_all_its_functions(tmp_path: Path) -> None:
    """Added files are new-side hunks, so every function in them is changed."""
    base = _init_repo(tmp_path)
    _write(tmp_path, "new.py", "def h():\n    return 1\n\n\ndef k():\n    return 2\n")
    _commit_all(tmp_path, "add new.py")
    assert _qualnames(tmp_path, base) == [("new.py", "h"), ("new.py", "k")]


def test_deleted_function_does_not_appear(tmp_path: Path) -> None:
    """Edge case: a removed ``def`` has no function left to report."""
    base = _init_repo(tmp_path)
    _write(tmp_path, "mod.py", "def f():\n    return 1\n")
    _commit_all(tmp_path, "drop g")
    assert changed(tmp_path, base) == ()


def test_deleted_file_reports_nothing(tmp_path: Path) -> None:
    """A deleted file holds no current functions, so it contributes nothing."""
    base = _init_repo(tmp_path)
    (tmp_path / "mod.py").unlink()
    _commit_all(tmp_path, "drop mod.py")
    assert changed(tmp_path, base) == ()


def test_missing_base_raises_tool_error(tmp_path: Path) -> None:
    """Edge case: ``base`` names no ref, so git fails and the error escapes as ``ToolError``."""
    _init_repo(tmp_path)
    with pytest.raises(ToolError):
        changed(tmp_path, "no-such-ref-12345")


def test_not_a_git_repository_raises_tool_error(tmp_path: Path) -> None:
    """Git failures other than a missing base are ``ToolError`` too."""
    _write(tmp_path, "mod.py", "def f():\n    return 1\n")
    with pytest.raises(ToolError):
        changed(tmp_path, "HEAD")


def test_no_changes_returns_empty(tmp_path: Path) -> None:
    """An empty diff reports no functions."""
    base = _init_repo(tmp_path)
    assert changed(tmp_path, base) == ()


def test_method_change_reports_the_method(tmp_path: Path) -> None:
    """A hunk inside ``C.m2`` reports ``C.m2`` but not ``C.m1``."""
    _init_repo(tmp_path)
    _write(
        tmp_path,
        "mod.py",
        "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
        "\n\nclass C:\n    def m1(self):\n        return 1\n\n    def m2(self):\n        return 2\n",
    )
    _commit_all(tmp_path, "add class")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _write(
        tmp_path,
        "mod.py",
        "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
        "\n\nclass C:\n    def m1(self):\n        return 1\n\n    def m2(self):\n        return 3\n",
    )
    _commit_all(tmp_path, "change m2")
    assert _qualnames(tmp_path, base) == [("mod.py", "C.m2")]


def test_two_hunks_report_both_functions(tmp_path: Path) -> None:
    """Two hunks in one commit report both touched functions, in file order."""
    base = _init_repo(tmp_path)
    _write(
        tmp_path,
        "mod.py",
        "def f():\n    return 10\n\n\ndef g():\n    return 2\n\n\ndef h():\n    return 30\n",
    )
    _commit_all(tmp_path, "change f and h")
    assert _qualnames(tmp_path, base) == [("mod.py", "f"), ("mod.py", "h")]
