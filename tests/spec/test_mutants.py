"""Unit tests for the rule-based mutants of ``pcc.spec.mutants`` (issue #36)."""

from __future__ import annotations

import ast

from pcc.spec.mutants import rule_mutants


def normalized(source: str) -> str:
    """The AST dump of ``source``, so formatting differences do not matter."""
    return ast.dump(ast.parse(source))


def assert_all_parse(mutants: tuple[str, ...]) -> None:
    for mutant in mutants:
        ast.parse(mutant)


def test_less_than_becomes_less_equal() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a < b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a <= b\n")]
    assert_all_parse(mutants)


def test_less_equal_becomes_less_than() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a <= b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a < b\n")]
    assert_all_parse(mutants)


def test_greater_than_becomes_greater_equal() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a > b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a >= b\n")]
    assert_all_parse(mutants)


def test_greater_equal_becomes_greater_than() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a >= b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a > b\n")]
    assert_all_parse(mutants)


def test_equal_becomes_not_equal() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a == b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a != b\n")]
    assert_all_parse(mutants)


def test_not_equal_becomes_equal() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a != b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a == b\n")]
    assert_all_parse(mutants)


def test_add_becomes_sub() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a + b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a - b\n")]
    assert_all_parse(mutants)


def test_sub_becomes_add() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a - b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a + b\n")]
    assert_all_parse(mutants)


def test_mul_becomes_add() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a * b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a + b\n")]
    assert_all_parse(mutants)


def test_int_constant_gives_plus_one_then_minus_one() -> None:
    mutants = rule_mutants("def f():\n    return 2\n", 10)
    assert [normalized(m) for m in mutants] == [
        normalized("def f():\n    return 3\n"),
        normalized("def f():\n    return 1\n"),
    ]
    assert_all_parse(mutants)


def test_negative_literal_mutates_the_constant() -> None:
    mutants = rule_mutants("def f():\n    return -5\n", 10)
    assert [normalized(m) for m in mutants] == [
        normalized("def f():\n    return -6\n"),
        normalized("def f():\n    return -4\n"),
    ]
    assert_all_parse(mutants)


def test_true_becomes_false() -> None:
    mutants = rule_mutants("def f():\n    return True\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f():\n    return False\n")]
    assert_all_parse(mutants)


def test_false_becomes_true() -> None:
    mutants = rule_mutants("def f():\n    return False\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f():\n    return True\n")]
    assert_all_parse(mutants)


def test_bool_constants_are_excluded_from_the_int_rule() -> None:
    mutants = rule_mutants("def f():\n    return True\n", 10)
    assert len(mutants) == 1
    assert_all_parse(mutants)


def test_and_becomes_or() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a and b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a or b\n")]
    assert_all_parse(mutants)


def test_or_becomes_and() -> None:
    mutants = rule_mutants("def f(a, b):\n    return a or b\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(a, b):\n    return a and b\n")]
    assert_all_parse(mutants)


def test_not_is_removed() -> None:
    mutants = rule_mutants("def f(x):\n    return not x\n", 10)
    assert [normalized(m) for m in mutants] == [normalized("def f(x):\n    return x\n")]
    assert_all_parse(mutants)


def test_len_comparison_example() -> None:
    source = "def f(s):\n    if len(s) <= 2:\n        return True\n    return False\n"
    mutants = rule_mutants(source, 10)
    dumps = [normalized(m) for m in mutants]
    assert normalized("def f(s):\n    if len(s) <= 1:\n        return True\n    return False\n") in dumps
    assert normalized("def f(s):\n    if len(s) <= 3:\n        return True\n    return False\n") in dumps
    assert normalized("def f(s):\n    if len(s) < 2:\n        return True\n    return False\n") in dumps
    assert_all_parse(mutants)


def test_limit_returns_the_first_mutants_in_source_order() -> None:
    source = (
        "def f(a, b, c, d, e):\n"
        "    x1 = a < b\n"
        "    x2 = c > d\n"
        "    x3 = a + b\n"
        "    x4 = c and d\n"
        "    x5 = e == c\n"
        "    return x1\n"
    )
    assert len(rule_mutants(source, 100)) == 5
    mutants = rule_mutants(source, 2)
    assert [normalized(m) for m in mutants] == [
        normalized(
            "def f(a, b, c, d, e):\n"
            "    x1 = a <= b\n"
            "    x2 = c > d\n"
            "    x3 = a + b\n"
            "    x4 = c and d\n"
            "    x5 = e == c\n"
            "    return x1\n"
        ),
        normalized(
            "def f(a, b, c, d, e):\n"
            "    x1 = a < b\n"
            "    x2 = c >= d\n"
            "    x3 = a + b\n"
            "    x4 = c and d\n"
            "    x5 = e == c\n"
            "    return x1\n"
        ),
    ]
    assert_all_parse(mutants)


def test_limit_zero_gives_no_mutants() -> None:
    assert rule_mutants("def f(a, b):\n    return a + b\n", 0) == ()


def test_function_without_candidate_gives_no_mutants() -> None:
    assert rule_mutants("def f(a):\n    return a\n", 10) == ()
    assert rule_mutants("def f():\n    pass\n", 10) == ()


def test_every_mutant_parses() -> None:
    sources = [
        "def f(s):\n    if len(s) <= 2:\n        return True\n    return False\n",
        "def f(a, b):\n    return a and b\n",
        "def f(x):\n    return not x\n",
        "def f(a, b):\n    return (a + 1) * (b - 2) if a != b else -3\n",
        "def f(a, b, c):\n    return a < b < c\n",
        "def f(a, b):\n    while a >= b and not a == b:\n        a = a - 1\n    return a or False\n",
    ]
    for source in sources:
        mutants = rule_mutants(source, 30)
        assert mutants, source
        assert_all_parse(mutants)
        for mutant in mutants:
            assert normalized(mutant) != normalized(source)
