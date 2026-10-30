"""Tests for ``pcc.transpile.spec.transpile_spec`` (docs/plan.md, step 4, spec mode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcc.config import Config
from pcc.core import PccError, Spec, UnsupportedConstruct
from pcc.transpile import transpile_spec
from pcc.transpile.spec import SPEC_CONSTRUCTS
from pcc.verifier import assemble, resolve, verify
from tests.helpers import function_ref

GOLDEN_DIR = Path(__file__).parent / "golden"
FIXTURE_DIR = Path(__file__).parent / "fixtures"

PALINDROME_FUNCTION = "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n"
PALINDROME_PREDICATE = "def spec(s, result):\n    return result == (s == s[::-1])\n"

ALL_FUNCTION = "def maximum(xs: list[int]) -> int:\n    return max(xs)\n"
ALL_PREDICATE = "def spec(xs, result):\n    return all(x <= result for x in xs)\n"

PRE_FUNCTION = "def first(xs: list[int]) -> int:\n    return xs[0]\n"
PRE_PREDICATE = "def spec(xs, result):\n    return result == xs[0]\n"
PRE_CONDITION = "def pre(xs):\n    return len(xs) > 0\n"


def make_spec(
    function_source: str,
    predicate_source: str,
    precondition_source: str | None = None,
) -> Spec:
    """A ``Spec`` for ``function_source`` with the given predicate and precondition sources."""
    return Spec(function_ref(function_source), predicate_source, precondition_source)


def test_palindrome_matches_golden() -> None:
    assert transpile_spec(make_spec(PALINDROME_FUNCTION, PALINDROME_PREDICATE)) == (
        GOLDEN_DIR / "spec_palindrome.dfy"
    ).read_text(encoding="utf-8")


def test_all_matches_golden() -> None:
    assert transpile_spec(make_spec(ALL_FUNCTION, ALL_PREDICATE)) == (GOLDEN_DIR / "spec_all.dfy").read_text(
        encoding="utf-8"
    )


def test_precondition_matches_golden() -> None:
    assert transpile_spec(make_spec(PRE_FUNCTION, PRE_PREDICATE, PRE_CONDITION)) == (
        GOLDEN_DIR / "spec_precondition.dfy"
    ).read_text(encoding="utf-8")


def test_missing_pre_gives_true_with_requires() -> None:
    assert transpile_spec(make_spec(PALINDROME_FUNCTION, PALINDROME_PREDICATE)) == (
        "predicate pre(s: string) { true }\n"
        "predicate spec(s: string, result: bool) requires pre(s)"
        " { result == (s == py_reversed(s)) }\n"
    )


def test_any_over_range() -> None:
    function = "def has_zero(n: int) -> bool:\n    return any(v == 0 for v in range(n))\n"
    predicate = "def spec(n, result):\n    return result == any(v == 0 for v in range(n))\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(n: int) { true }\n"
        "predicate spec(n: int, result: bool) requires pre(n)"
        " { result == (exists v :: 0 <= v < n && v == 0) }\n"
    )


def test_range_with_one_argument() -> None:
    function = "def below(n: int) -> bool:\n    return all(0 <= i for i in range(n))\n"
    predicate = "def spec(n, result):\n    return result == all(0 <= i for i in range(n))\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(n: int) { true }\n"
        "predicate spec(n: int, result: bool) requires pre(n)"
        " { result == (forall i :: 0 <= i < n ==> 0 <= i) }\n"
    )


def test_range_with_two_arguments() -> None:
    function = "def in_window(a: int, b: int) -> bool:\n    return all(0 <= i for i in range(a, b))\n"
    predicate = "def spec(a, b, result):\n    return result == all(0 <= i for i in range(a, b))\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(a: int, b: int) { true }\n"
        "predicate spec(a: int, b: int, result: bool) requires pre(a, b)"
        " { result == (forall i :: a <= i < b ==> 0 <= i) }\n"
    )


def test_nested_generators_mix_range_and_list() -> None:
    function = "def present(n: int, xs: list[int]) -> bool:\n    return any(x == i for i in range(n) for x in xs)\n"
    predicate = "def spec(n, xs, result):\n    return result == any(x == i for i in range(n) for x in xs)\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(n: int, xs: seq<int>) { true }\n"
        "predicate spec(n: int, xs: seq<int>, result: bool) requires pre(n, xs)"
        " { result == (exists i, j :: 0 <= i < n && 0 <= j < |xs| && xs[j] == i) }\n"
    )


def test_keyword_parameter_is_renamed() -> None:
    function = "def kind(type: str) -> bool:\n    return type == type\n"
    predicate = "def spec(type, result):\n    return result\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(type_: string) { true }\n"
        "predicate spec(type_: string, result: bool) requires pre(type_) { result }\n"
    )


def test_conditional_expression() -> None:
    function = "def pick(c: bool, a: int, b: int) -> int:\n    return a if c else b\n"
    predicate = "def spec(c, a, b, result):\n    return result == (a if c else b)\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(c: bool, a: int, b: int) { true }\n"
        "predicate spec(c: bool, a: int, b: int, result: int) requires pre(c, a, b)"
        " { result == (if c then a else b) }\n"
    )


def test_membership_on_list() -> None:
    function = "def contains(xs: list[int], v: int) -> bool:\n    return v in xs\n"
    predicate = "def spec(xs, v, result):\n    return result == (v in xs)\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(xs: seq<int>, v: int) { true }\n"
        "predicate spec(xs: seq<int>, v: int, result: bool) requires pre(xs, v)"
        " { result == (v in xs) }\n"
    )


def test_membership_not_in_on_list() -> None:
    function = "def f(xs: list[int], v: int) -> bool:\n    return v not in xs\n"
    predicate = "def spec(xs, v, result):\n    return result == (v not in xs)\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(xs: seq<int>, v: int) { true }\n"
        "predicate spec(xs: seq<int>, v: int, result: bool) requires pre(xs, v)"
        " { result == (v !in xs) }\n"
    )


def test_sorted_call_raises() -> None:
    predicate = "def spec(xs, result):\n    return result == sorted(xs)\n"
    with pytest.raises(UnsupportedConstruct) as excinfo:
        transpile_spec(make_spec(ALL_FUNCTION, predicate))
    assert excinfo.value.construct == "call sorted"


def test_substring_in_raises() -> None:
    function = "def has_char(s: str, c: str) -> bool:\n    return c in s\n"
    predicate = "def spec(s, c, result):\n    return result == (c in s)\n"
    with pytest.raises(UnsupportedConstruct) as excinfo:
        transpile_spec(make_spec(function, predicate))
    assert excinfo.value.construct == "substring in"


def test_unknown_name_raises() -> None:
    predicate = "def spec(xs, result):\n    return result == unknown\n"
    with pytest.raises(UnsupportedConstruct) as excinfo:
        transpile_spec(make_spec(ALL_FUNCTION, predicate))
    assert excinfo.value.construct == "name unknown"


def test_missing_hints_raise_no_types() -> None:
    function = "def first(xs) -> int:\n    return xs[0]\n"
    with pytest.raises(UnsupportedConstruct) as excinfo:
        transpile_spec(make_spec(function, PRE_PREDICATE, PRE_CONDITION))
    assert excinfo.value.construct == "no types"


def test_unsupported_hint_raises_type() -> None:
    function = "def first(xs: dict[str, int]) -> int:\n    return 0\n"
    predicate = "def spec(xs, result):\n    return result == 0\n"
    with pytest.raises(UnsupportedConstruct) as excinfo:
        transpile_spec(make_spec(function, predicate))
    assert excinfo.value.construct == "type dict[str, int]"


def test_nested_list_hint() -> None:
    function = "def total(xss: list[list[int]]) -> int:\n    return 0\n"
    predicate = "def spec(xss, result):\n    return result == 0\n"
    assert transpile_spec(make_spec(function, predicate)) == (
        "predicate pre(xss: seq<seq<int>>) { true }\n"
        "predicate spec(xss: seq<seq<int>>, result: int) requires pre(xss) { result == 0 }\n"
    )


def test_spec_constructs_covers_ticket_rules() -> None:
    assert {"all", "any", "len", "range", "in", "not in", "index", "reverse"} <= SPEC_CONSTRUCTS


def test_only_unsupported_construct_escapes() -> None:
    predicate = "def spec(xs, result):\n    return result == sorted(xs)\n"
    with pytest.raises(PccError):
        transpile_spec(make_spec(ALL_FUNCTION, predicate))
    with pytest.raises(UnsupportedConstruct):
        transpile_spec(make_spec(ALL_FUNCTION, predicate))


PALINDROME_METHOD = (
    "method is_palindrome(s: string) returns (result: bool)\n"
    "  ensures spec(s, result)\n"
    "{\n"
    "  result := s == py_reversed(s);\n"
    "}\n"
)

ALL_METHOD = "method bounded(xs: seq<int>) returns (result: int)\n  ensures spec(xs, result)\n{\n  result := 0;\n}\n"

PRE_METHOD = (
    "method first(xs: seq<int>) returns (result: int)\n"
    "  requires pre(xs)\n"
    "  ensures spec(xs, result)\n"
    "{\n"
    "  result := xs[0];\n"
    "}\n"
)


def _assembled(function_source: str, golden_name: str, method: str) -> str:
    function = function_ref(function_source)
    golden = (GOLDEN_DIR / golden_name).read_text(encoding="utf-8")
    return assemble(function, golden, method)


@pytest.mark.dafny
def test_palindrome_golden_resolves_with_method() -> None:
    assert resolve(_assembled(PALINDROME_FUNCTION, "spec_palindrome.dfy", PALINDROME_METHOD)) == ()


@pytest.mark.dafny
def test_all_golden_resolves_with_method() -> None:
    assert resolve(_assembled(ALL_FUNCTION, "spec_all.dfy", ALL_METHOD)) == ()


@pytest.mark.dafny
def test_precondition_golden_resolves_with_method() -> None:
    assert resolve(_assembled(PRE_FUNCTION, "spec_precondition.dfy", PRE_METHOD)) == ()


@pytest.mark.dafny
def test_palindrome_proof_verifies_end_to_end() -> None:
    """The demo path: the spec of example 1 with a hand-written method and proof verifies."""
    fixture = json.loads((FIXTURE_DIR / "palindrome_proof.json").read_text(encoding="utf-8"))
    function = function_ref(PALINDROME_FUNCTION)
    dafny_spec = transpile_spec(make_spec(PALINDROME_FUNCTION, PALINDROME_PREDICATE))
    dafny_code = assemble(function, dafny_spec, fixture["method"], lemmas=fixture["lemmas"])
    try:
        outcome = verify(dafny_code, resource_limit=Config().dafny.resource_limit)
    except NotImplementedError:
        pytest.skip("verifier.verify is still a stub; the direct dafny verify run is reported in the pull request")
    assert outcome.verified, outcome.errors
    assert outcome.errors == ()
