"""Unit tests for the read-only tools of the context search agent (issue #32)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pcc.context import tools
from pcc.core.errors import ToolError
from tests.helpers import function_ref

FUNCTION_SOURCE = "def maximum(xs: list[int]) -> int:\n    return max(xs)\n"


def make_repo(tmp_path: Path, source: str = FUNCTION_SOURCE) -> tuple[Path, object]:
    """A repo with the function defined in ``src/shop.py`` at line 3."""
    package = tmp_path / "src" / "shop.py"
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_text('"""Shop helpers."""\n\n' + source)
    function = function_ref(source=source, path="src/shop.py", qualname="maximum", lineno=3)
    return tmp_path, function


def test_read_file_returns_at_most_200_lines(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    big = tmp_path / "src" / "big.py"
    big.write_text("".join(f"line {index}\n" for index in range(250)))
    result = tools.read_file(repo_root, "src/big.py", function)  # type: ignore[arg-type]
    assert len(result.splitlines()) <= 200


def test_read_file_masks_the_function_body(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    result = tools.read_file(repo_root, "src/shop.py", function)  # type: ignore[arg-type]
    assert FUNCTION_SOURCE not in result
    assert "return max(xs)" not in result
    assert tools.OMITTED in result


def test_read_file_keeps_unrelated_files_intact(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    other = tmp_path / "src" / "other.py"
    other.write_text("def helper() -> int:\n    return 1\n")
    result = tools.read_file(repo_root, "src/other.py", function)  # type: ignore[arg-type]
    assert "return 1" in result
    assert tools.OMITTED not in result


def test_read_file_outside_repo_returns_an_error(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    result = tools.read_file(repo_root, "../secret", function)  # type: ignore[arg-type]
    assert "error" in result.lower()
    assert FUNCTION_SOURCE not in result


def test_read_file_absolute_path_outside_repo_returns_an_error(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    result = tools.read_file(repo_root, "/etc/passwd", function)  # type: ignore[arg-type]
    assert "error" in result.lower()


def test_read_file_missing_file_returns_an_error(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    result = tools.read_file(repo_root, "src/missing.py", function)  # type: ignore[arg-type]
    assert "error" in result.lower()


def test_grep_returns_at_most_50_matches(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    haystack = tmp_path / "src" / "hay.py"
    haystack.write_text("".join("needle here\n" for _ in range(60)))
    result = tools.grep(repo_root, "needle", function)  # type: ignore[arg-type]
    assert len(result.splitlines()) <= 50


def test_grep_masks_body_lines_of_the_function_file(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    result = tools.grep(repo_root, "max", function)  # type: ignore[arg-type]
    assert FUNCTION_SOURCE not in result
    assert "return max(xs)" not in result


def test_grep_finds_matches_in_other_files(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    caller = tmp_path / "src" / "cart.py"
    caller.write_text("from shop import maximum\n\ntotal = maximum([1, 2])\n")
    result = tools.grep(repo_root, "maximum", function)  # type: ignore[arg-type]
    assert "src/cart.py" in result
    assert FUNCTION_SOURCE not in result


def test_grep_invalid_regex_returns_an_error(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    result = tools.grep(repo_root, "(unclosed", function)  # type: ignore[arg-type]
    assert "error" in result.lower()


def test_git_log_returns_at_most_20_commits(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True)
    target = repo_root / "src" / "shop.py"
    for index in range(25):
        target.write_text(target.read_text() + f"# change {index}\n")
        subprocess.run(["git", "add", "src/shop.py"], cwd=repo_root, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"change {index}"], cwd=repo_root, check=True, capture_output=True)
    result = tools.git_log(repo_root, "src/shop.py", function)  # type: ignore[arg-type]
    assert len(result.splitlines()) <= 20
    assert FUNCTION_SOURCE not in result


def test_git_log_outside_repo_returns_an_error(tmp_path: Path) -> None:
    repo_root, function = make_repo(tmp_path)
    result = tools.git_log(repo_root, "../secret", function)  # type: ignore[arg-type]
    assert "error" in result.lower()


def test_missing_repo_root_raises_tool_error(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-repo"
    function = function_ref(source=FUNCTION_SOURCE, path="src/shop.py", qualname="maximum", lineno=3)
    try:
        tools.read_file(missing, "src/shop.py", function)  # type: ignore[arg-type]
    except ToolError:
        pass
    else:
        raise AssertionError("expected ToolError")
