"""Tests for ``pcc.transpile.code``: tier 1 lists and loops (issue #44).

Covers ``for``/``while`` loops, ``+=``, tuple assignment, subscript stores on local
lists, shared-list rejection, list comprehensions, ``min``/``sum`` and calls to
project functions, plus the golden ``z_function``/``go_next``/``find_pattern`` tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from pcc.core import UnsupportedConstruct
from pcc.transpile.code import transpile_function
from pcc.verifier import resolve
from tests.helpers import function_ref

GOLDEN_DIR = Path(__file__).parent / "golden"

GO_NEXT_SOURCE = (
    "def go_next(s: str, i: int, z: list[int]) -> bool:\n    return i + z[i] < len(s) and s[z[i]] == s[i + z[i]]\n"
)

Z_FUNCTION_SOURCE = (
    "def z_function(s: str) -> list[int]:\n"
    "    n = len(s)\n"
    "    z = [0 for i in range(n)]\n"
    "    l, r = 0, 0\n"
    "    for i in range(1, n):\n"
    "        if i <= r:\n"
    "            z[i] = min(r - i + 1, z[i - l])\n"
    "        while go_next(s, i, z):\n"
    "            z[i] += 1\n"
    "        if i + z[i] - 1 > r:\n"
    "            l, r = i, i + z[i] - 1\n"
    "    return z\n"
)

FIND_PATTERN_FIXED_SOURCE = (
    "def find_pattern(pattern: str, text: str) -> int:\n"
    '    combined = pattern + "$" + text\n'
    "    z_result = z_function(combined)\n"
    "    n = len(pattern)\n"
    "    return sum(value >= n for value in z_result[n:])\n"
)

FIND_PATTERN_BUGGY_SOURCE = (
    "def find_pattern(pattern: str, text: str) -> int:\n"
    '    combined = pattern + "$" + text\n'
    "    z_result = z_function(combined)\n"
    "    n = len(pattern)\n"
    "    answer = 0\n"
    "    for value in z_result[n:]:\n"
    "        if value >= n:\n"
    "            answer += 1\n"
    "    return answer\n"
)


def transpile(
    source: str,
    qualname: str = "f",
    path: str = "src/mod.py",
    callees: Mapping[str, str] | None = None,
    read_only_callees: frozenset[str] = frozenset(),
    callee_types: Mapping[str, str] | None = None,
) -> str:
    """Transpile ``source`` with project-call info, defaulting to no callees."""
    function = function_ref(source=source, path=path, qualname=qualname)
    return transpile_function(
        function,
        callees=callees if callees is not None else MappingProxyType({}),
        read_only_callees=read_only_callees,
        callee_types=callee_types if callee_types is not None else MappingProxyType({}),
    )


def check_unsupported(
    source: str,
    construct: str,
    lineno: int = 2,
    qualname: str = "f",
    callees: Mapping[str, str] | None = None,
    read_only_callees: frozenset[str] = frozenset(),
    callee_types: Mapping[str, str] | None = None,
) -> None:
    """Transpiling ``source`` raises ``UnsupportedConstruct(construct)`` at ``lineno``."""
    with pytest.raises(UnsupportedConstruct) as exc_info:
        transpile(
            source,
            qualname=qualname,
            callees=callees,
            read_only_callees=read_only_callees,
            callee_types=callee_types,
        )
    assert exc_info.value.construct == construct
    assert exc_info.value.lineno == lineno


def golden_module(name: str) -> str:
    """The golden file ``name``: a complete module with hand-written specs around the methods."""
    return (GOLDEN_DIR / name).read_text(encoding="utf-8")


def golden_method(name: str, method: str) -> None:
    """The transpiled ``method`` appears verbatim (indented by two spaces) in the golden file."""
    indented = "\n".join(f"  {line}" if line else line for line in method.split("\n"))
    assert f"\n{indented}\n" in golden_module(name)


def golden_without_include(name: str) -> str:
    """The golden file without its ``include`` line, for ``verifier.resolve``."""
    kept = [line for line in golden_module(name).splitlines() if not line.startswith("include")]
    return "\n".join(kept) + "\n"


def test_read_only_callees_defaults_to_empty() -> None:
    function = function_ref(source="def f(a: int) -> int:\n    return a + 1\n")
    out = transpile_function(function)
    assert "return a + 1;" in out


def test_for_range_two_bounds_uses_counter_loop() -> None:
    source = (
        "def f(n: int) -> int:\n    total = 0\n    for i in range(1, n):\n        total = total + i\n    return total\n"
    )
    out = transpile(source)
    assert "  var i := 1;\n  var pcc_t1 := n;\n  while i < pcc_t1\n  {\n" in out
    assert "    total := total + i;\n    i := i + 1;\n  }\n  return total;\n" in out


def test_for_range_single_bound_starts_at_zero() -> None:
    source = (
        "def f(n: int) -> int:\n    total = 0\n    for i in range(n):\n        total = total + i\n    return total\n"
    )
    out = transpile(source)
    assert "  var i := 0;\n  var pcc_t1 := n;\n  while i < pcc_t1\n  {\n" in out


def test_for_range_reuses_declared_counter() -> None:
    source = "def f(n: int) -> int:\n    i = 0\n    for i in range(n):\n        i = i\n    return i\n"
    out = transpile(source)
    assert "  var i := 0;\n  i := 0;\n  var pcc_t1 := n;\n  while i < pcc_t1\n  {\n" in out
    assert "    i := i;\n    i := i + 1;\n" in out


def test_for_range_declared_counter_restarts_at_lower() -> None:
    source = "def f(n: int) -> int:\n    i = 0\n    for i in range(1, n):\n        i = i\n    return i\n"
    out = transpile(source)
    assert "  var i := 0;\n  i := 1;\n  var pcc_t1 := n;\n  while i < pcc_t1\n  {\n" in out


def test_for_range_counter_type_mismatch() -> None:
    check_unsupported(
        "def f(s: str, n: int) -> int:\n    for s in range(n):\n        s = s\n    return n\n",
        "assignment type",
        lineno=2,
    )


def test_range_with_step_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> int:\n    for i in range(0, n, 2):\n        n = n\n    return n\n",
        "range step",
        lineno=2,
    )


def test_range_arity_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> int:\n    for i in range():\n        n = n\n    return n\n",
        "range arguments",
        lineno=2,
    )


def test_range_non_int_bound_rejected() -> None:
    check_unsupported(
        'def f(n: int) -> int:\n    for i in range("x"):\n        n = n\n    return n\n',
        "range arguments",
        lineno=2,
    )


def test_for_else_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> int:\n    for i in range(n):\n        n = n\n    else:\n        n = 0\n    return n\n",
        "loop else",
        lineno=2,
    )


def test_for_over_list_uses_index_loop() -> None:
    source = (
        "def f(xs: list[int]) -> int:\n"
        "    total = 0\n"
        "    for value in xs:\n"
        "        total = total + value\n"
        "    return total\n"
    )
    out = transpile(source)
    assert "  var pcc_t1 := xs;\n  var pcc_t2 := 0;\n  while pcc_t2 < py_len(pcc_t1)\n  {\n" in out
    assert "    var value := py_index(pcc_t1, pcc_t2);\n" in out
    assert "    total := total + value;\n    pcc_t2 := pcc_t2 + 1;\n  }\n  return total;\n" in out


def test_for_over_slice() -> None:
    source = (
        "def f(xs: list[int], n: int) -> int:\n"
        "    total = 0\n"
        "    for value in xs[n:]:\n"
        "        total = total + value\n"
        "    return total\n"
    )
    out = transpile(source)
    assert "  var pcc_t1 := py_slice(xs, n, py_len(xs));\n" in out


def test_for_over_string_rejected() -> None:
    check_unsupported(
        "def f(s: str) -> int:\n    for c in s:\n        s = s\n    return 0\n",
        "for iterable",
        lineno=2,
    )


def test_for_over_int_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> int:\n    for i in n:\n        n = n\n    return n\n",
        "for iterable",
        lineno=2,
    )


def test_for_non_name_target_rejected() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> int:\n    for xs[0] in xs:\n        xs = xs\n    return 0\n",
        "for target",
        lineno=2,
    )


def test_for_over_list_mutated_in_body_is_shared() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> int:\n    for v in xs:\n        xs[0] = 1\n    return 0\n",
        "shared list",
        lineno=3,
    )


def test_for_over_list_augassign_in_body_is_shared() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> int:\n    for v in xs:\n        xs[0] += 1\n    return 0\n",
        "shared list",
        lineno=3,
    )


def test_for_over_slice_mutation_stays_allowed() -> None:
    source = (
        "def f(n: int) -> list[int]:\n"
        "    xs = [0 for i in range(n)]\n"
        "    for v in xs[0:]:\n"
        "        xs[0] = 1\n"
        "    return xs\n"
    )
    out = transpile(source)
    assert "xs := py_set_index(xs, 0, 1);" in out


def test_while_without_calls_stays_plain() -> None:
    source = "def f(n: int) -> int:\n    while n > 0:\n        n = n - 1\n    return n\n"
    out = transpile(source)
    assert "  while n > 0\n  {\n    n := n - 1;\n  }\n  return n;\n" in out


def test_while_with_len_stays_inline() -> None:
    source = "def f(s: str, n: int) -> int:\n    while n < len(s):\n        n = n + 1\n    return n\n"
    out = transpile(source)
    assert "  while n < py_len(s)\n  {\n" in out


def test_while_single_call_hoisted_once_per_iteration() -> None:
    source = "def f(n: int) -> bool:\n    while f(n):\n        n = n\n    return True\n"
    out = transpile(source)
    assert "  while true\n  {\n    var pcc_c := f(n);\n    if !pcc_c { break; }\n    n := n;\n  }\n" in out


def test_while_project_call_hoisted() -> None:
    source = "def f(n: int) -> bool:\n    while go(n):\n        n = n\n    return True\n"
    out = transpile(source, callees={"go": "GoldenGo"}, callee_types={"go": "bool"}, read_only_callees=frozenset())
    assert "    var pcc_c := GoldenGo.go(n);\n    if !pcc_c { break; }\n" in out


def test_while_project_call_bool_default_without_callee_types() -> None:
    source = "def f(n: int) -> bool:\n    while go(n):\n        n = n\n    return True\n"
    out = transpile(source, callees={"go": "GoldenGo"})
    assert "    var pcc_c := GoldenGo.go(n);\n    if !pcc_c { break; }\n" in out


def test_while_and_chain_short_circuits() -> None:
    source = "def f(x: bool) -> bool:\n    while f(x) and x:\n        x = f(x)\n    return x\n"
    out = transpile(source)
    assert "    var pcc_c := f(x);\n    if !pcc_c { break; }\n    if !x { break; }\n" in out


def test_while_or_chain_short_circuits() -> None:
    source = "def f(x: bool) -> bool:\n    while f(x) or x:\n        x = x\n    return x\n"
    out = transpile(source)
    assert "    var pcc_c := f(x);\n    if !pcc_c {\n      if !x { break; }\n    }\n" in out


def test_while_nested_boolop_with_call_rejected() -> None:
    check_unsupported(
        "def f(x: bool, y: bool) -> bool:\n    while (x or f(y)) and y:\n        x = y\n    return x\n",
        "while condition",
        lineno=2,
    )


def test_while_ifexp_with_call_rejected() -> None:
    check_unsupported(
        "def f(x: bool, y: bool) -> bool:\n    while (x if f(y) else y):\n        x = y\n    return x\n",
        "while condition",
        lineno=2,
    )


def test_while_non_bool_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> int:\n    while n:\n        n = n - 1\n    return n\n",
        "non-bool condition",
        lineno=2,
    )


def test_while_else_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> int:\n    while n > 0:\n        n = n - 1\n    else:\n        n = 0\n    return n\n",
        "loop else",
        lineno=2,
    )


def test_int_augassign() -> None:
    out = transpile("def f(a: int) -> int:\n    a += 1\n    return a\n")
    assert "  a := a + 1;\n  return a;\n" in out


def test_augassign_undeclared_rejected() -> None:
    check_unsupported("def f(a: int) -> int:\n    x += 1\n    return a\n", "augmented assignment", lineno=2)


def test_augassign_str_rejected() -> None:
    check_unsupported('def f(s: str) -> str:\n    s += "x"\n    return s\n', "augmented assignment", lineno=2)


def test_tuple_first_assignment_declares() -> None:
    out = transpile("def f(a: int, b: int) -> int:\n    x, y = a, b\n    return x\n")
    assert "  var x, y := a, b;\n  return x;\n" in out


def test_tuple_reassignment_has_no_var() -> None:
    out = transpile("def f(a: int, b: int) -> int:\n    x, y = a, a\n    x, y = b, b\n    return x\n")
    assert "  var x, y := a, a;\n  x, y := b, b;\n  return x;\n" in out


def test_tuple_mixed_declaration_rejected() -> None:
    check_unsupported(
        "def f(a: int, b: int) -> int:\n    x = a\n    x, y = a, b\n    return x\n",
        "tuple assignment",
        lineno=3,
    )


def test_tuple_type_mismatch_rejected() -> None:
    check_unsupported(
        'def f(a: int) -> int:\n    x, y = a, "s"\n    return x\n',
        "assignment type",
        lineno=2,
    )


def test_subscript_store_on_local_list() -> None:
    source = "def f(n: int) -> list[int]:\n    xs = [0 for i in range(n)]\n    xs[0] = 1\n    return xs\n"
    out = transpile(source)
    assert "  xs := py_set_index(xs, 0, 1);\n  return xs;\n" in out


def test_subscript_store_negative_index() -> None:
    source = "def f(n: int) -> list[int]:\n    xs = [0 for i in range(n)]\n    xs[-1] = 1\n    return xs\n"
    out = transpile(source)
    assert "  xs := py_set_index(xs, -1, 1);\n  return xs;\n" in out


def test_subscript_store_on_param_is_shared() -> None:
    check_unsupported("def f(xs: list[int]) -> int:\n    xs[0] = 1\n    return xs[0]\n", "shared list", lineno=2)


def test_subscript_store_type_mismatch() -> None:
    check_unsupported(
        'def f(n: int) -> list[int]:\n    xs = [0 for i in range(n)]\n    xs[0] = "s"\n    return xs\n',
        "assignment type",
        lineno=3,
    )


def test_subscript_augassign_on_local_list() -> None:
    source = "def f(n: int) -> list[int]:\n    xs = [0 for i in range(n)]\n    xs[0] += 2\n    return xs\n"
    out = transpile(source)
    assert "  xs := py_set_index(xs, 0, py_index(xs, 0) + 2);\n  return xs;\n" in out


def test_aliasing_a_list_is_shared() -> None:
    check_unsupported("def f(xs: list[int]) -> list[int]:\n    ys = xs\n    return ys\n", "shared list", lineno=2)


def test_alias_then_mutate_is_shared() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> list[int]:\n    ys = xs\n    ys[0] = 1\n    return ys\n",
        "shared list",
        lineno=2,
    )


def test_list_passed_to_uncertified_callee_is_shared() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> int:\n    return g(xs)\n",
        "shared list",
        lineno=2,
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )


def test_list_passed_to_read_only_callee_is_allowed() -> None:
    out = transpile(
        "def f(xs: list[int]) -> int:\n    return g(xs)\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
        read_only_callees=frozenset({"g"}),
    )
    assert "  var pcc_t1 := GoldenG.g(xs);\n  return pcc_t1;\n" in out


def test_list_passed_to_self_call_is_allowed() -> None:
    out = transpile("def f(xs: list[int]) -> int:\n    return f(xs[1:])\n")
    assert "  var pcc_t1 := f(py_slice(xs, 1, py_len(xs)));\n  return pcc_t1;\n" in out


def test_str_passed_to_uncertified_callee_is_allowed() -> None:
    out = transpile(
        "def f(s: str) -> int:\n    return g(s)\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )
    assert "  var pcc_t1 := GoldenG.g(s);\n  return pcc_t1;\n" in out


def test_project_call_in_return_hoisted() -> None:
    out = transpile(
        "def f(a: int) -> int:\n    return g(a)\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )
    assert "  var pcc_t1 := GoldenG.g(a);\n  return pcc_t1;\n" in out


def test_project_call_composite_assign_hoisted() -> None:
    out = transpile(
        "def f(a: int) -> int:\n    x = g(a) + 1\n    return x\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )
    assert "  var pcc_t1 := GoldenG.g(a);\n  var x := pcc_t1 + 1;\n  return x;\n" in out
    assert "GoldenG.g(a) + 1" not in out


def test_project_call_composite_return_hoisted() -> None:
    out = transpile(
        "def f(a: int) -> int:\n    return g(a) + 1\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )
    assert "  var pcc_t1 := GoldenG.g(a);\n  return pcc_t1 + 1;\n" in out


def test_project_call_in_if_hoisted() -> None:
    out = transpile(
        "def f(a: int) -> int:\n    x = 0\n    if g(a):\n        x = 1\n    return x\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "bool"},
    )
    assert "  var pcc_t1 := GoldenG.g(a);\n" in out
    assert "  if pcc_t1 {\n" in out


def test_tuple_with_project_call_hoisted() -> None:
    out = transpile(
        "def f(a: int) -> int:\n    x, y = g(a), 1\n    return x\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )
    assert "  var pcc_t1 := GoldenG.g(a);\n" in out
    assert "  var x, y := pcc_t1, 1;\n  return x;\n" in out


def test_subscript_store_with_project_call_hoisted() -> None:
    out = transpile(
        "def f(n: int, a: int) -> list[int]:\n    xs = [0 for i in range(n)]\n    xs[0] = g(a)\n    return xs\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )
    assert "  var pcc_t1 := GoldenG.g(a);\n" in out
    assert "  xs := py_set_index(xs, 0, pcc_t1);\n  return xs;\n" in out


def test_augassign_with_project_call_hoisted() -> None:
    out = transpile(
        "def f(a: int) -> int:\n    x = 0\n    x += g(a)\n    return x\n",
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )
    assert "  var pcc_t1 := GoldenG.g(a);\n" in out
    assert "  x := x + pcc_t1;\n  return x;\n" in out


def test_ambiguous_project_call_rejected() -> None:
    check_unsupported(
        "def f(a: int) -> int:\n    return g(a)\n",
        "call g",
        lineno=2,
        callees={"pkg.g": "M1", "other.g": "M2"},
        callee_types={"pkg.g": "int", "other.g": "int"},
    )


def test_project_call_without_type_in_value_position() -> None:
    check_unsupported(
        "def f(a: int) -> int:\n    return g(a) + 1\n",
        "unknown type",
        lineno=2,
        callees={"g": "GoldenG"},
    )


def test_project_call_result_binds_mutable_list() -> None:
    source = "def f(s: str) -> list[int]:\n    z = g(s)\n    z[0] = 1\n    return z\n"
    out = transpile(source, callees={"g": "GoldenG"}, callee_types={"g": "seq<int>"})
    assert "  var z := GoldenG.g(s);\n  z := py_set_index(z, 0, 1);\n  return z;\n" in out


def test_nested_project_call_in_return_short_circuit() -> None:
    source = "def f(a: int, b: bool) -> bool:\n    return b or g(a)\n"
    out = transpile(source, callees={"g": "GoldenG"}, callee_types={"g": "bool"})
    assert "  if b {\n    return true;\n  } else {\n    var pcc_t1 := GoldenG.g(a);\n    return pcc_t1;\n  }\n" in out


def test_comprehension_basic() -> None:
    out = transpile("def f(n: int) -> list[int]:\n    return [0 for i in range(n)]\n")
    assert out.endswith("{\n  return seq(n, i => 0);\n}")


def test_comprehension_uses_loop_var() -> None:
    out = transpile("def f(n: int) -> list[int]:\n    return [i * 2 for i in range(n)]\n")
    assert out.endswith("{\n  return seq(n, i => i * 2);\n}")


def test_comprehension_with_call_bound() -> None:
    out = transpile("def f(s: str) -> list[int]:\n    return [0 for i in range(len(s))]\n")
    assert out.endswith("{\n  return seq(py_len(s), i => 0);\n}")


def test_comprehension_filter_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> list[int]:\n    return [i for i in range(n) if i > 0]\n",
        "comprehension",
        lineno=2,
    )


def test_comprehension_non_range_rejected() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> list[int]:\n    return [x for x in xs]\n",
        "comprehension",
        lineno=2,
    )


def test_comprehension_two_generators_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> list[int]:\n    return [i for i in range(n) for j in range(n)]\n",
        "comprehension",
        lineno=2,
    )


def test_comprehension_with_project_call_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> list[int]:\n    return [g(i) for i in range(n)]\n",
        "call in comprehension",
        lineno=2,
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )


def test_comprehension_with_self_call_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> list[int]:\n    return [f(i) for i in range(n)]\n",
        "call in comprehension",
        lineno=2,
    )


def test_comprehension_with_nested_project_call_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> list[int]:\n    return [g(i) + 1 for i in range(n)]\n",
        "call in comprehension",
        lineno=2,
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )


def test_comprehension_with_project_call_in_assign_rejected() -> None:
    check_unsupported(
        "def f(n: int) -> list[int]:\n    xs = [g(i) for i in range(n)]\n    return xs\n",
        "call in comprehension",
        lineno=2,
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )


def test_min_two_ints() -> None:
    out = transpile("def f(a: int, b: int) -> int:\n    return min(a, b)\n")
    assert out.endswith("{\n  return py_min(a, b);\n}")


def test_min_arity_rejected() -> None:
    check_unsupported("def f(a: int, b: int, c: int) -> int:\n    return min(a, b, c)\n", "min arguments", lineno=2)


def test_min_non_int_rejected() -> None:
    check_unsupported("def f(a: str, b: str) -> str:\n    return min(a, b)\n", "min arguments", lineno=2)


def test_min_keyword_rejected() -> None:
    check_unsupported("def f(a: int, b: int) -> int:\n    return min(a, b, default=0)\n", "min arguments", lineno=2)


def test_sum_bool_generator() -> None:
    out = transpile("def f(xs: list[int], n: int) -> int:\n    return sum(v >= n for v in xs)\n")
    assert out.endswith("{\n  return py_count(xs, v => v >= n);\n}")


def test_sum_over_slice() -> None:
    out = transpile("def f(xs: list[int], n: int) -> int:\n    return sum(v >= n for v in xs[n:])\n")
    assert out.endswith("{\n  return py_count(py_slice(xs, n, py_len(xs)), v => v >= n);\n}")


def test_sum_non_bool_rejected() -> None:
    check_unsupported("def f(xs: list[int]) -> int:\n    return sum(v for v in xs)\n", "sum generator", lineno=2)


def test_sum_with_filter_rejected() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> int:\n    return sum(v for v in xs if v > 0)\n",
        "sum generator",
        lineno=2,
    )


def test_sum_non_sequence_rejected() -> None:
    check_unsupported("def f(s: str) -> int:\n    return sum(v for v in s)\n", "sum generator", lineno=2)


def test_sum_with_project_call_rejected() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> int:\n    return sum(g(v) for v in xs)\n",
        "call in sum",
        lineno=2,
        callees={"g": "GoldenG"},
        callee_types={"g": "bool"},
    )


def test_sum_with_self_call_rejected() -> None:
    check_unsupported(
        "def f(xs: list[int]) -> bool:\n    return sum(f(xs) for v in xs)\n",
        "call in sum",
        lineno=2,
    )


def test_sum_with_nested_project_call_rejected() -> None:
    check_unsupported(
        "def f(xs: list[int], n: int) -> int:\n    return sum(v + g(v) >= n for v in xs)\n",
        "call in sum",
        lineno=2,
        callees={"g": "GoldenG"},
        callee_types={"g": "int"},
    )


def test_callee_tag_ignores_short_name_collision() -> None:
    check_unsupported(
        "def f(a: int) -> int:\n    return g(a) + 1\n",
        "unknown type",
        lineno=2,
        callees={"g": "GoldenG"},
        callee_types={"pkg.g": "int", "other.g": "bool"},
    )


def test_callee_tag_exact_qualname_match() -> None:
    out = transpile(
        "def f(a: int) -> int:\n    return g(a) + 1\n",
        callees={"pkg.g": "GoldenG"},
        callee_types={"pkg.g": "int"},
    )
    assert "  var pcc_t1 := GoldenG.g(a);\n  return pcc_t1 + 1;\n" in out


def test_go_next_matches_golden() -> None:
    method = transpile(GO_NEXT_SOURCE, qualname="go_next", path="src/z_function.py")
    golden_method("go_next.dfy", method)


def test_z_function_matches_golden() -> None:
    method = transpile(
        Z_FUNCTION_SOURCE,
        qualname="z_function",
        path="src/z_function.py",
        callees={"go_next": "GoldenGoNext"},
        read_only_callees=frozenset({"go_next"}),
    )
    golden_method("z_function.dfy", method)


def test_find_pattern_fixed_matches_golden() -> None:
    method = transpile(
        FIND_PATTERN_FIXED_SOURCE,
        qualname="find_pattern",
        path="src/z_function.py",
        callees={"z_function": "GoldenZFunction"},
        callee_types={"z_function": "seq<int>"},
    )
    golden_method("find_pattern_fixed.dfy", method)


def test_find_pattern_buggy_matches_golden() -> None:
    method = transpile(
        FIND_PATTERN_BUGGY_SOURCE,
        qualname="find_pattern",
        path="src/z_function.py",
        callees={"z_function": "GoldenZFunction"},
        callee_types={"z_function": "seq<int>"},
    )
    golden_method("find_pattern_buggy.dfy", method)


@pytest.mark.dafny
def test_golden_go_next_resolves_with_hand_written_spec() -> None:
    assert resolve(golden_without_include("go_next.dfy")) == ()


@pytest.mark.dafny
def test_golden_z_function_resolves_with_hand_written_spec() -> None:
    assert resolve(golden_without_include("z_function.dfy")) == ()


@pytest.mark.dafny
def test_golden_find_pattern_fixed_resolves_with_hand_written_spec() -> None:
    assert resolve(golden_without_include("find_pattern_fixed.dfy")) == ()


@pytest.mark.dafny
def test_golden_find_pattern_buggy_resolves_with_hand_written_spec() -> None:
    assert resolve(golden_without_include("find_pattern_buggy.dfy")) == ()
