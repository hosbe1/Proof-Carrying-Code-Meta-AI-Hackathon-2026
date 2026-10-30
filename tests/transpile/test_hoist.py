"""Tests for ``pcc.transpile.code``: hoisting recursive calls (issue #27)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcc.core import UnsupportedConstruct
from pcc.transpile.code import transpile_function
from pcc.verifier import method_name, resolve
from tests.helpers import function_ref

GOLDEN_DIR = Path(__file__).parent / "golden"

BUGGY_SOURCE = (
    "def is_palindrome_recursive(s: str) -> bool:\n"
    "    if len(s) <= 2:\n"
    "        return True\n"
    "    if s[0] != s[-1]:\n"
    "        return False\n"
    "    return is_palindrome_recursive(s[1:-1])\n"
)

FIXED_SOURCE = (
    "def is_palindrome_recursive(s: str) -> bool:\n"
    "    if len(s) <= 1:\n"
    "        return True\n"
    "    if s[0] != s[-1]:\n"
    "        return False\n"
    "    return is_palindrome_recursive(s[1:-1])\n"
)


def transpile(source: str, qualname: str = "f", path: str = "src/mod.py") -> str:
    """Transpile ``source`` as ``qualname`` in ``path``."""
    function = function_ref(source=source, path=path, qualname=qualname)
    return transpile_function(function)


def check_unsupported(source: str, construct: str, lineno: int = 2, qualname: str = "f") -> None:
    """Transpiling ``source`` raises ``UnsupportedConstruct(construct)`` at ``lineno``."""
    function = function_ref(source=source, path="src/mod.py", qualname=qualname)
    with pytest.raises(UnsupportedConstruct) as exc_info:
        transpile_function(function)
    assert exc_info.value.construct == construct
    assert exc_info.value.lineno == lineno


def golden_module(name: str) -> str:
    """The golden file ``name``: a complete module with a hand-written spec around the method."""
    return (GOLDEN_DIR / name).read_text(encoding="utf-8")


def golden_method(name: str, method: str) -> None:
    """The transpiled ``method`` appears verbatim (indented by two spaces) in the golden file."""
    indented = "\n".join(f"  {line}" if line else line for line in method.split("\n"))
    assert f"\n{indented}\n" in golden_module(name)


def golden_without_include(name: str) -> str:
    """The golden file without its ``include`` line, for ``verifier.resolve``."""
    kept = [line for line in golden_module(name).splitlines() if not line.startswith("include")]
    return "\n".join(kept) + "\n"


def test_recursive_call_hoisted_before_return() -> None:
    out = transpile("def f(s: str) -> str:\n    return f(s[1:-1])\n")
    assert "  var pcc_t1 := f(py_slice(s, 1, -1));\n  return pcc_t1;\n" in out


def test_hoisted_name_uses_pcc_prefix() -> None:
    out = transpile("def f(s: str) -> str:\n    return f(s)\n")
    assert "pcc_t1" in out
    assert "_t1" not in out.replace("pcc_t1", "")


def test_two_calls_hoisted_left_to_right() -> None:
    out = transpile("def f(n: int) -> int:\n    return f(n - 1) + f(n - 1)\n")
    assert "  var pcc_t1 := f(n - 1);\n  var pcc_t2 := f(n - 1);\n  return pcc_t1 + pcc_t2;\n" in out


def test_callee_uses_method_name() -> None:
    function = function_ref(
        source="def inner(n: int) -> int:\n    return inner(n - 1)\n",
        path="src/mod.py",
        qualname="outer.inner",
    )
    out = transpile_function(function)
    assert "method outer_inner(n: int)" in out
    assert "var pcc_t1 := outer_inner(n - 1);" in out
    assert out.startswith(f"method {method_name(function)}(n: int)")


def test_shortcircuit_and_expanded_under_return() -> None:
    out = transpile("def f(n: int) -> bool:\n    return n > 0 and f(n - 1)\n")
    assert "  if n > 0 {\n    var pcc_t1 := f(n - 1);\n    return pcc_t1;\n  } else {\n    return false;\n  }\n" in out


def test_shortcircuit_or_example() -> None:
    out = transpile("def f(n: int) -> bool:\n    return n <= 0 or f(n - 1)\n")
    assert "  if n <= 0 {\n    return true;\n  } else {\n    var pcc_t1 := f(n - 1);\n    return pcc_t1;\n  }\n" in out


def test_shortcircuit_ternary_expanded_under_return() -> None:
    out = transpile("def f(n: int) -> int:\n    return f(n - 1) if n > 0 else 0\n")
    assert "  if n > 0 {\n    var pcc_t1 := f(n - 1);\n    return pcc_t1;\n  } else {\n    return 0;\n  }\n" in out


def test_assignment_hoists_when_no_shortcircuit() -> None:
    out = transpile("def f(n: int) -> int:\n    x = f(n - 1)\n    return x\n")
    assert "  var pcc_t1 := f(n - 1);\n  var x := pcc_t1;\n  return x;\n" in out


def test_if_condition_hoists_when_no_shortcircuit() -> None:
    out = transpile("def f(n: int) -> int:\n    if f(n - 1) == 0:\n        return 0\n    return 1\n")
    assert "  var pcc_t1 := f(n - 1);\n  if pcc_t1 == 0 {\n    return 0;\n  }\n" in out


def test_call_in_shortcircuit_elsewhere_raises() -> None:
    check_unsupported(
        "def f(y: bool, z: bool) -> bool:\n    x = f(y, z) and z\n    return x\n",
        "call in short-circuit",
        lineno=2,
    )


def test_call_in_or_elsewhere_raises() -> None:
    check_unsupported(
        "def f(y: bool, z: bool) -> bool:\n    x = y or f(y, z)\n    return x\n",
        "call in short-circuit",
        lineno=2,
    )


def test_call_in_conditional_branch_elsewhere_raises() -> None:
    check_unsupported(
        "def f(n: int) -> int:\n    x = f(n - 1) if n > 0 else 0\n    return x\n",
        "call in short-circuit",
        lineno=2,
    )


def test_call_in_if_condition_shortcircuit_raises() -> None:
    check_unsupported(
        "def f(n: int) -> bool:\n    if n > 0 and f(n - 1):\n        return True\n    return False\n",
        "call in short-circuit",
        lineno=2,
    )


def test_call_in_shortcircuit_nested_in_binop_under_return_raises() -> None:
    check_unsupported(
        "def f(n: int) -> int:\n    return f(n - 1) + (1 if n > 0 else f(n - 1))\n",
        "call in short-circuit",
        lineno=2,
    )


def test_nested_or_under_and_raises() -> None:
    check_unsupported(
        "def f(x: bool, y: bool) -> bool:\n    return (x or f(x, y)) and y\n",
        "call in short-circuit",
        lineno=2,
    )


def test_nested_and_under_or_raises() -> None:
    check_unsupported(
        "def f(x: bool, y: bool) -> bool:\n    return (x and f(x, y)) or y\n",
        "call in short-circuit",
        lineno=2,
    )


def test_nested_boolop_in_ifexp_condition_raises() -> None:
    check_unsupported(
        "def f(x: bool, y: bool) -> bool:\n    return True if (x and f(x, y)) else False\n",
        "call in short-circuit",
        lineno=2,
    )


def test_other_project_call_still_rejected() -> None:
    check_unsupported("def f(a: int) -> int:\n    return other(a)\n", "call other", lineno=2)


def test_header_has_exactly_one_requires_and_one_ensures() -> None:
    out = transpile("def f(n: int) -> bool:\n    return n <= 0 or f(n - 1)\n")
    header = out.split("{\n")[0].splitlines()
    assert header[0] == "method f(n: int) returns (result: bool)"
    requires = [line for line in header if line.strip().startswith("requires")]
    ensures = [line for line in header if line.strip().startswith("ensures")]
    assert requires == ["  requires pre(n)"]
    assert ensures == ["  ensures spec(n, result)"]


def test_buggy_palindrome_matches_golden() -> None:
    method = transpile(BUGGY_SOURCE, qualname="is_palindrome_recursive")
    assert "py_len" in method
    assert "py_index" in method
    assert "py_slice(s, 1, -1)" in method
    golden_method("palindrome_buggy.dfy", method)


def test_fixed_palindrome_matches_golden() -> None:
    method = transpile(FIXED_SOURCE, qualname="is_palindrome_recursive")
    assert "py_len" in method
    assert "py_index" in method
    assert "py_slice(s, 1, -1)" in method
    golden_method("palindrome_fixed.dfy", method)


@pytest.mark.dafny
def test_golden_buggy_resolves_with_hand_written_spec() -> None:
    assert resolve(golden_without_include("palindrome_buggy.dfy")) == ()


@pytest.mark.dafny
def test_golden_fixed_resolves_with_hand_written_spec() -> None:
    assert resolve(golden_without_include("palindrome_fixed.dfy")) == ()
