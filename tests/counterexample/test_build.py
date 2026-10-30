"""Step 6, build: re-running a failing input and packaging it as a ready-to-run pytest."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pcc.core import FunctionRef, Spec, ToolError
from pcc.counterexample import build
from pcc.sandbox import run_python
from tests.helpers import function_ref

PALINDROME_SOURCE = (
    "def is_palindrome(s: str) -> bool:\n"
    "    if len(s) <= 2:\n"
    "        return True\n"
    "    return s[0] == s[-1] and is_palindrome(s[1:-1])\n"
)
PALINDROME_PREDICATE = "def spec(s, result):\n    return result == (s == s[::-1])\n"

ADD_SOURCE = "def add(a: int, b: int) -> int:\n    return a - b\n"
ADD_PREDICATE = "def spec(a, b, result):\n    return result == a + b\n"

RAISE_SOURCE = 'def boom(x: int) -> int:\n    raise ValueError("nope")\n'
RAISE_PREDICATE = "def spec(x, result):\n    return result == x\n"

PICK_SOURCE = (
    "def pick(xs: list[int], key: int | None) -> int:\n    if key is None:\n        return xs[0]\n    return xs[key]\n"
)
PICK_PREDICATE = "def spec(xs, result, key=None):\n    return result == (xs[0] if key is None else xs[key])\n"


def write_module(repo_root: Path, relative: str, source: str) -> None:
    """Write ``source`` to ``repo_root / relative``, creating parent packages with ``__init__.py``."""
    path = repo_root / relative
    for parent in path.parents:
        if parent == repo_root:
            break
        parent.mkdir(parents=True, exist_ok=True)
        init = parent / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    path.write_text(source, encoding="utf-8")


def make_spec(function: FunctionRef, predicate: str) -> Spec:
    """A spec for ``function`` with ``predicate`` and no precondition."""
    return Spec(function=function, predicate=predicate)


def run_test_file(repo_root: Path, name: str, test_source: str) -> int:
    """Write ``test_source`` to ``name`` under ``repo_root`` and run it with pytest in the sandbox."""
    (repo_root / name).write_text(test_source, encoding="utf-8")
    script = f"import pytest\nimport sys\nsys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', {name!r}]))\n"
    return run_python(script, cwd=repo_root).returncode


def test_buggy_palindrome_builds_and_generated_test_fails(tmp_path: Path) -> None:
    write_module(tmp_path, "strings/palindrome.py", PALINDROME_SOURCE)
    function = function_ref(source=PALINDROME_SOURCE, path="strings/palindrome.py", qualname="is_palindrome", lineno=1)
    found = build(function, make_spec(function, PALINDROME_PREDICATE), tmp_path, "'abca'")

    assert found.arguments == "'abca'"
    assert found.actual == "True"
    assert "\n" not in found.expected
    assert "def spec(s, result):" in found.expected

    assert run_test_file(tmp_path, "test_palindrome_breaks.py", found.test) != 0


def test_multiple_parameters_use_positional_spec_call(tmp_path: Path) -> None:
    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    found = build(function, make_spec(function, ADD_PREDICATE), tmp_path, "3, 4")

    assert found.actual == "-1"
    assert "assert spec(3, 4, result)" in found.test


def test_raising_function_reports_the_exception_and_test_errors(tmp_path: Path) -> None:
    write_module(tmp_path, "boomer.py", RAISE_SOURCE)
    function = function_ref(source=RAISE_SOURCE, path="boomer.py", qualname="boom", lineno=1)
    found = build(function, make_spec(function, RAISE_PREDICATE), tmp_path, "1")

    assert found.actual == "ValueError('nope')"
    assert run_test_file(tmp_path, "test_boom_breaks.py", found.test) != 0


def test_keyword_arguments_stay_valid_python(tmp_path: Path) -> None:
    write_module(tmp_path, "picker.py", PICK_SOURCE)
    function = function_ref(source=PICK_SOURCE, path="picker.py", qualname="pick", lineno=1)
    found = build(function, make_spec(function, PICK_PREDICATE), tmp_path, "[3, -1], key=None")

    assert found.actual == "3"
    ast.parse(found.test)
    assert "**_kwargs" in found.test


ANSWER_SOURCE = "def answer() -> int:\n    return 41\n"
ANSWER_PREDICATE = "def spec(result):\n    return result == 42\n"


def test_no_arguments_call_without_parentheses_content(tmp_path: Path) -> None:
    write_module(tmp_path, "answer.py", ANSWER_SOURCE)
    function = function_ref(source=ANSWER_SOURCE, path="answer.py", qualname="answer", lineno=1)
    found = build(function, make_spec(function, ANSWER_PREDICATE), tmp_path, "")

    assert found.arguments == ""
    assert found.actual == "41"
    assert "assert spec(result)" in found.test
    assert run_test_file(tmp_path, "test_answer_breaks.py", found.test) != 0


def test_missing_repo_root_raises_tool_error(tmp_path: Path) -> None:
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    with pytest.raises(ToolError):
        build(function, make_spec(function, ADD_PREDICATE), tmp_path / "missing", "3, 4")
