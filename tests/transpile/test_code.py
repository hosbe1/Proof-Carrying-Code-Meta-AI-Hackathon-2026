"""Tests for ``pcc.transpile.code``: code mode types, expressions and statements."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pcc.core import FunctionRef, UnsupportedConstruct
from pcc.transpile.code import translate_expr, transpile_function
from pcc.verifier import method_name, resolve
from tests.helpers import function_ref

GOLDEN_DIR = Path(__file__).parent / "golden"


def make_source(body: str, header: str = "def f(a: int, b: int) -> int:") -> str:
    """A one-function source with ``body`` indented under ``header``."""
    return "".join(f"{header}\n    {line}\n" if line else "\n" for line in body.split("\n"))


def transpile(source: str, qualname: str = "f", path: str = "src/mod.py") -> str:
    """Transpile ``source`` as ``qualname`` in ``path``."""
    function = function_ref(source=source, path=path, qualname=qualname)
    return transpile_function(function)


def check_unsupported(source: str, construct: str, lineno: int = 1) -> None:
    """Transpiling ``source`` raises ``UnsupportedConstruct(construct)`` at ``lineno``."""
    with pytest.raises(UnsupportedConstruct) as exc_info:
        transpile(source)
    assert exc_info.value.construct == construct
    assert exc_info.value.lineno == lineno


def header_of(source: str) -> list[str]:
    """The header lines (up to the opening brace) of the transpiled ``source``."""
    return transpile(source).split("{\n")[0].splitlines()


def golden_module(name: str) -> str:
    """The golden file ``name``: a complete module with a hand-written spec around the method."""
    return (GOLDEN_DIR / name).read_text(encoding="utf-8")


def golden_method(name: str, method: str) -> None:
    """The transpiled ``method`` appears verbatim (indented by two spaces) in the golden file."""
    indented = "\n".join(f"  {line}" if line else line for line in method.split("\n"))
    assert f"\n{indented}\n" in golden_module(name)


def golden_without_include(name: str) -> str:
    """The golden file without its ``include`` line, for ``verifier.resolve``.

    ``resolve`` passes ``PyLib.dfy`` on the Dafny command line instead, like
    ``verifier.assemble`` output which carries no ``include`` line either.
    """
    kept = [line for line in golden_module(name).splitlines() if not line.startswith("include")]
    return "\n".join(kept) + "\n"


def test_example1_floordiv_matches_golden() -> None:
    source = "def add(a: int, b: int) -> int:\n    return a // b\n"
    method = transpile(source, qualname="add")
    assert "return py_floordiv(a, b);" in method
    golden_method("add_floordiv.dfy", method)


def test_example3_same_ends_matches_golden() -> None:
    source = "def same(s: str) -> bool:\n    return s[0] == s[-1]\n"
    method = transpile(source, qualname="same")
    assert "return py_index(s, 0) == py_index(s, -1);" in method
    golden_method("same_ends.dfy", method)


@pytest.mark.dafny
def test_golden_add_floordiv_resolves_with_hand_written_spec() -> None:
    assert resolve(golden_without_include("add_floordiv.dfy")) == ()


@pytest.mark.dafny
def test_golden_same_ends_resolves_with_hand_written_spec() -> None:
    assert resolve(golden_without_include("same_ends.dfy")) == ()


def test_example4_unary_minus() -> None:
    assert transpile("def neg(x: int) -> int:\n    return -x\n", qualname="neg").endswith("{\n  return -x;\n}")


def test_header_has_exactly_one_requires_and_one_ensures() -> None:
    header = header_of("def add(a: int, b: int) -> int:\n    return a + b\n")
    assert header[0] == "method f(a: int, b: int) returns (result: int)"
    requires = [line for line in header if line.strip().startswith("requires")]
    ensures = [line for line in header if line.strip().startswith("ensures")]
    assert requires == ["  requires pre(a, b)"]
    assert ensures == ["  ensures spec(a, b, result)"]


def test_method_name_uses_layout() -> None:
    function = function_ref(source="def m(x: int) -> int:\n    return x\n", qualname="Cart.total")
    assert transpile_function(function).startswith(f"method {method_name(function)}(x: int)")


def test_method_name_for_nested_qualname() -> None:
    function = function_ref(source="def m(x: int) -> int:\n    return x\n", qualname="outer.inner")
    assert transpile_function(function).startswith("method outer_inner(x: int)")


def test_literals() -> None:
    assert transpile("def f() -> int:\n    return 42\n").endswith("{\n  return 42;\n}")
    assert transpile("def f() -> bool:\n    return True\n").endswith("{\n  return true;\n}")
    assert transpile("def f() -> bool:\n    return False\n").endswith("{\n  return false;\n}")
    assert transpile('def f() -> str:\n    return "hi"\n').endswith('{\n  return "hi";\n}')


def test_string_escapes() -> None:
    out = transpile('def f() -> str:\n    return "a\\"b\\\\c"\n')
    assert out.endswith('{\n  return "a\\"b\\\\c";\n}')


def test_int_arithmetic() -> None:
    assert "return a + b;" in transpile(make_source("return a + b"))
    assert "return a - b;" in transpile(make_source("return a - b"))
    assert "return a * b;" in transpile(make_source("return a * b"))
    assert "return py_floordiv(a, b);" in transpile(make_source("return a // b"))
    assert "return py_mod(a, b);" in transpile(make_source("return a % b"))


def test_int_comparisons() -> None:
    for op in ("<", "<=", ">", ">=", "==", "!="):
        source = f"def f(a: int, b: int) -> bool:\n    return a {op} b\n"
        assert f"return a {op} b;" in transpile(source)


def test_equality_on_str_and_list() -> None:
    out = transpile("def f(s: str, t: str) -> bool:\n    return s == t\n")
    assert out.endswith("{\n  return s == t;\n}")
    out = transpile("def f(s: str, t: str) -> bool:\n    return s != t\n")
    assert out.endswith("{\n  return s != t;\n}")
    out = transpile("def f(xs: list[int], ys: list[int]) -> bool:\n    return xs == ys\n")
    assert out.endswith("{\n  return xs == ys;\n}")


def test_bool_logic() -> None:
    out = transpile("def f(a: bool, b: bool) -> bool:\n    return a and b\n")
    assert out.endswith("{\n  return a && b;\n}")
    out = transpile("def f(a: bool, b: bool) -> bool:\n    return a or b\n")
    assert out.endswith("{\n  return a || b;\n}")
    out = transpile("def f(a: bool) -> bool:\n    return not a\n")
    assert out.endswith("{\n  return !a;\n}")


def test_len_index_slice_reversed() -> None:
    out = transpile("def f(s: str) -> int:\n    return len(s)\n")
    assert out.endswith("{\n  return py_len(s);\n}")
    out = transpile("def f(xs: list[int]) -> int:\n    return len(xs)\n")
    assert out.endswith("{\n  return py_len(xs);\n}")
    out = transpile("def f(xs: list[int]) -> int:\n    return xs[2]\n")
    assert out.endswith("{\n  return py_index(xs, 2);\n}")
    out = transpile("def f(s: str) -> str:\n    return s[1:-1]\n")
    assert out.endswith("{\n  return py_slice(s, 1, -1);\n}")
    out = transpile("def f(s: str) -> str:\n    return s[1:]\n")
    assert out.endswith("{\n  return py_slice(s, 1, py_len(s));\n}")
    out = transpile("def f(s: str) -> str:\n    return s[:2]\n")
    assert out.endswith("{\n  return py_slice(s, 0, 2);\n}")
    out = transpile("def f(s: str) -> str:\n    return s[::-1]\n")
    assert out.endswith("{\n  return py_reversed(s);\n}")


def test_char_equality() -> None:
    out = transpile("def same(s: str) -> bool:\n    return s[0] == s[-1]\n", qualname="same")
    assert out.endswith("{\n  return py_index(s, 0) == py_index(s, -1);\n}")


def test_concat_on_str_and_list() -> None:
    out = transpile("def f(s: str, t: str) -> str:\n    return s + t\n")
    assert out.endswith("{\n  return s + t;\n}")
    out = transpile("def f(xs: list[int], ys: list[int]) -> list[int]:\n    return xs + ys\n")
    assert out.endswith("{\n  return xs + ys;\n}")


def test_ternary() -> None:
    out = transpile("def f(a: bool, b: int, c: int) -> int:\n    return b if a else c\n")
    assert out.endswith("{\n  return if a then b else c;\n}")


def test_if_elif_else() -> None:
    source = (
        "def f(x: int) -> int:\n"
        "    if x > 0:\n"
        "        return 1\n"
        "    elif x < 0:\n"
        "        return -1\n"
        "    else:\n"
        "        return 0\n"
    )
    out = transpile(source)
    assert "  if x > 0 {\n    return 1;\n  } else if x < 0 {\n    return -1;\n  } else {\n    return 0;\n  }\n" in out


def test_if_without_else_followed_by_return() -> None:
    source = "def f(x: int) -> int:\n    if x > 0:\n        return 1\n    return 0\n"
    out = transpile(source)
    assert "  if x > 0 {\n    return 1;\n  }\n  return 0;\n" in out


def test_assignment_uses_var_first_then_plain() -> None:
    source = "def f(a: int) -> int:\n    x = a + 1\n    x = x + 1\n    return x\n"
    out = transpile(source)
    assert "  var x := a + 1;\n  x := x + 1;\n  return x;\n" in out


def test_assignment_to_parameter_has_no_var() -> None:
    source = "def f(a: int) -> int:\n    a = a + 1\n    return a\n"
    out = transpile(source)
    assert "  a := a + 1;\n" in out


def test_docstring_is_skipped() -> None:
    source = 'def f(a: int) -> int:\n    """Add one."""\n    return a + 1\n'
    assert transpile(source).endswith("{\n  return a + 1;\n}")


def test_renamed_reserved_parameter_used_everywhere() -> None:
    source = "def f(seq: list[int]) -> int:\n    return len(seq)\n"
    out = transpile(source)
    assert "method f(seq_: seq<int>) returns (result: int)" in out
    assert "  requires pre(seq_)" in out
    assert "  ensures spec(seq_, result)" in out
    assert out.endswith("{\n  return py_len(seq_);\n}")


def test_renamed_result_and_locals() -> None:
    source = "def f(result: int, var: int) -> int:\n    type = result + var\n    return type\n"
    out = transpile(source)
    assert "method f(result_: int, var_: int)" in out
    assert "  var type_ := result_ + var_;\n  return type_;\n" in out


def test_translate_expr_uses_rename_map() -> None:
    node = ast.parse("seq + 1", mode="eval").body
    assert translate_expr(node, renames={"seq": "seq_"}, types={"seq": "int"}) == "seq_ + 1"


def test_translate_expr_optional_types_for_spec_reuse() -> None:
    node = ast.parse("a + 1", mode="eval").body
    with pytest.raises(UnsupportedConstruct) as exc_info:
        translate_expr(node, renames={})
    assert exc_info.value.construct == "unknown type"
    assert translate_expr(node, renames={}, types={"a": "int"}) == "a + 1"


def test_translate_expr_unknown_type_raises() -> None:
    node = ast.parse("a + b", mode="eval").body
    with pytest.raises(UnsupportedConstruct) as exc_info:
        translate_expr(node, renames={})
    assert exc_info.value.construct == "unknown type"


def test_callees_lower_project_calls() -> None:
    function = function_ref(source="def f(a: int) -> int:\n    return a + 1\n", path="src/mod.py", qualname="f")
    out = transpile_function(function, callees={"other": "PCC_other"})
    assert "return a + 1;" in out
    typed = function_ref(source="def f(a: int) -> int:\n    return other(a)\n", path="src/mod.py", qualname="f")
    out = transpile_function(typed, callees={"other": "PCC_other"}, callee_types={"other": "int"})
    assert "return PCC_other.other(a);" in out
    check_unsupported("def f(a: int) -> int:\n    return unknown(a)\n", "call unknown", lineno=2)


def test_example2_str_comparison() -> None:
    check_unsupported('def f(s: str) -> bool: return s < "b"\n', "str comparison <")


def test_str_comparison_all_operators() -> None:
    for op in ("<=", ">", ">="):
        source = f'def f(s: str) -> bool:\n    return s {op} "b"\n'
        check_unsupported(source, f"str comparison {op}", lineno=2)


def test_char_vs_str() -> None:
    check_unsupported('def same(s: str) -> bool:\n    return s[0] == "a"\n', "char vs str", lineno=2)
    check_unsupported('def same(s: str) -> bool:\n    return s[0] != "a"\n', "char vs str", lineno=2)


def test_bool_arithmetic_rejected() -> None:
    check_unsupported("def f(a: bool) -> int:\n    return True + 1\n", "bool arithmetic", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    return a + True\n", "bool arithmetic", lineno=2)


def test_lineno_relative_to_source_with_decorator() -> None:
    source = '@deco\ndef f(s: str) -> bool:\n    return s < "b"\n'
    function = function_ref(source=source, path="src/mod.py", qualname="f")
    with pytest.raises(UnsupportedConstruct) as exc_info:
        transpile_function(function)
    assert exc_info.value.construct == "str comparison <"
    assert exc_info.value.lineno == 3


def test_lineno_of_second_line() -> None:
    source = "def f(a: int) -> int:\n    x = 1.5\n    return a\n"
    check_unsupported(source, "float literal", lineno=2)


def test_missing_hints() -> None:
    check_unsupported("def f(a) -> int:\n    return a\n", "no types")
    check_unsupported("def f(a: int):\n    return a\n", "no types")


def test_type_errors() -> None:
    check_unsupported("def f(a: tuple[int, str]) -> int:\n    return 0\n", "type tuple")
    check_unsupported("def f(a: float) -> int:\n    return 0\n", "type float")
    check_unsupported("def f(a: int) -> None:\n    return a\n", "type None")


def test_mutated_list() -> None:
    check_unsupported("def f(xs: list[int]) -> int:\n    xs[0] = 1\n    return xs[0]\n", "shared list", lineno=2)
    check_unsupported("def f(xs: list[int]) -> int:\n    xs[0:1] = [2]\n    return xs[0]\n", "mutated list", lineno=2)
    check_unsupported("def f(xs: list[int]) -> list[int]:\n    xs += [1]\n    return xs\n", "mutated list", lineno=2)
    check_unsupported("def f(xs: list[int]) -> list[int]:\n    xs.append(1)\n    return xs\n", "mutated list", lineno=2)
    check_unsupported("def f(xs: list[int]) -> list[int]:\n    ys = xs\n    return ys\n", "shared list", lineno=2)


def test_mutating_method_names() -> None:
    for name in ("extend", "insert", "pop", "remove", "clear", "sort", "reverse"):
        source = f"def f(xs: list[int]) -> list[int]:\n    xs.{name}(1)\n    return xs\n"
        check_unsupported(source, "mutated list", lineno=2)


def test_surrogate_rejected() -> None:
    check_unsupported('def f() -> str:\n    return "\\ud800"\n', "lone surrogate", lineno=2)


def test_non_bool_logic() -> None:
    check_unsupported("def f(a: int) -> int:\n    return a and 1\n", "non-bool logic", lineno=2)
    check_unsupported("def f(a: int) -> bool:\n    return not a\n", "non-bool logic", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    return 1 if a else 2\n", "non-bool condition", lineno=2)


def test_unsupported_operators() -> None:
    source = "def f(a: int) -> int:\n    return a / 2\n"
    check_unsupported(source, "operator /", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    return a ** 2\n", "operator **", lineno=2)
    check_unsupported("def f(a: int) -> bool:\n    return a in [1]\n", "operator in", lineno=2)


def test_unsupported_calls() -> None:
    check_unsupported("def f(a: int) -> int:\n    return g(a)\n", "call g", lineno=2)
    check_unsupported("def f(s: str) -> str:\n    return s.upper()\n", "call upper", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    return abs(a)\n", "call abs", lineno=2)


def test_len_operand() -> None:
    check_unsupported("def f(a: int) -> int:\n    return len(a)\n", "len operand", lineno=2)


def test_index_type() -> None:
    check_unsupported('def f(a: int) -> int:\n    return a["x"]\n', "index type", lineno=2)
    check_unsupported('def f(s: str) -> str:\n    return s["x"]\n', "index type", lineno=2)


def test_slice_step() -> None:
    check_unsupported("def f(s: str) -> str:\n    return s[::2]\n", "slice step", lineno=2)


def test_bool_and_char_comparisons() -> None:
    check_unsupported("def f(a: bool) -> bool:\n    return a < True\n", "bool comparison", lineno=2)
    check_unsupported("def f(s: str) -> bool:\n    return s[0] < s[1]\n", "char comparison", lineno=2)


def test_unsupported_statements() -> None:
    check_unsupported("def f(a: int) -> int:\n    break\n    return a\n", "break", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    continue\n    return a\n", "continue", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    raise ValueError\n    return a\n", "raise", lineno=2)
    check_unsupported(
        "def f(a: int) -> int:\n    try:\n        return a\n    except ValueError:\n        return 0\n",
        "try",
        lineno=2,
    )
    check_unsupported("def f(a: int) -> int:\n    with open('x') as fh:\n        return a\n", "with", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    x = yield a\n    return x\n", "yield", lineno=2)
    check_unsupported("async def f(a: int) -> int:\n    return a\n", "async function", lineno=1)
    check_unsupported("def f(a: int) -> int:\n    global g\n    return a\n", "global", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    nonlocal g\n    return a\n", "nonlocal", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    assert a > 0\n    return a\n", "assert", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    x = y = a\n    return x\n", "multiple assignment", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    x, y = a\n    return x\n", "tuple assignment", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    x += 1\n    return a\n", "augmented assignment", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    x: int\n    return a\n", "bare annotation", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    ...\n", "ellipsis", lineno=2)
    check_unsupported(
        "def f(a: int) -> int:\n    def g(b: int) -> int:\n        return b\n    return g(a)\n",
        "nested function",
        lineno=2,
    )
    check_unsupported("def f(a: int) -> int:\n    g = lambda b: b\n    return a\n", "lambda", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    return {1: 2}[1]\n", "dict literal", lineno=2)
    check_unsupported('def f(a: int) -> str:\n    return f"{a}"\n', "f-string", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    del a\n    return 0\n", "del", lineno=2)
    check_unsupported("def f(a: int, *args: int) -> int:\n    return a\n", "variadic", lineno=1)


def test_missing_and_mismatched_return() -> None:
    check_unsupported("def f(a: int) -> int:\n    x = a + 1\n", "missing return", lineno=2)
    check_unsupported("def f(a: int) -> int:\n    return\n", "empty return", lineno=2)
    check_unsupported('def f(a: int) -> int:\n    return "x"\n', "return type", lineno=2)


def test_only_unsupported_construct_escapes() -> None:
    function: FunctionRef = function_ref(source="def f(a: int) -> int:\n    return a\n")
    assert isinstance(transpile_function(function), str)
