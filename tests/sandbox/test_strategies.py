"""Hypothesis strategies built from the type hints of the function under test."""

from __future__ import annotations

import math
import unicodedata
from typing import Any

import pytest
from hypothesis import given, settings

from pcc.core import UnsupportedConstruct
from pcc.sandbox.strategies import strategies_source
from tests.helpers import function_ref


def draws(source: str, count: int = 100) -> list[Any]:
    """Execute ``source`` and draw ``count`` argument tuples from its ``args_strategy``."""
    namespace: dict[str, Any] = {}
    exec(source, namespace)
    strategy = namespace["args_strategy"]
    seen: list[Any] = []

    @settings(max_examples=count)
    @given(strategy)
    def collect(args: Any) -> None:
        seen.append(args)

    collect()
    return seen


def test_str_and_int_parameters_draw_tuples() -> None:
    function = function_ref("def f(s: str, n: int) -> bool:\n    return True\n")
    values = draws(strategies_source(function))
    assert len(values) == 100
    for args in values:
        assert isinstance(args, tuple) and len(args) == 2
        assert isinstance(args[0], str)
        assert isinstance(args[1], int)


def test_typing_list_draws_same_as_builtin_list() -> None:
    for hint in ("typing.List[int]", "list[int]"):
        function = function_ref(f"def f(xs: {hint}) -> int:\n    return 0\n")
        values = draws(strategies_source(function))
        assert len(values) == 100
        for args in values:
            assert isinstance(args, tuple) and len(args) == 1
            assert isinstance(args[0], list)
            assert len(args[0]) <= 20
            assert all(isinstance(item, int) for item in args[0])


def test_float_draws_are_finite() -> None:
    function = function_ref("def f(x: float) -> float:\n    return x\n")
    values = draws(strategies_source(function))
    assert len(values) == 100
    for args in values:
        assert isinstance(args, tuple) and len(args) == 1
        assert isinstance(args[0], float)
        assert math.isfinite(args[0])


def test_optional_spelling_draws_none_and_ints() -> None:
    function = function_ref("def f(x: Optional[int]) -> int:\n    return 0\n")
    values = draws(strategies_source(function))
    assert any(args[0] is None for args in values)
    assert any(isinstance(args[0], int) for args in values)


def test_pipe_none_draws_none_and_ints() -> None:
    function = function_ref("def f(x: int | None) -> int:\n    return 0\n")
    values = draws(strategies_source(function))
    assert any(args[0] is None for args in values)
    assert any(isinstance(args[0], int) for args in values)


def test_nested_list_draws_nested_lists() -> None:
    function = function_ref("def f(xs: list[list[int]]) -> int:\n    return 0\n")
    values = draws(strategies_source(function))
    assert len(values) == 100
    for args in values:
        assert isinstance(args, tuple) and len(args) == 1
        assert isinstance(args[0], list)
        for inner in args[0]:
            assert isinstance(inner, list)
            assert all(isinstance(item, int) for item in inner)


def test_tuple_pair_draws_pairs() -> None:
    for hint in ("tuple[str, int]", "typing.Tuple[str, int]"):
        function = function_ref(f"def f(pair: {hint}) -> bool:\n    return True\n")
        values = draws(strategies_source(function))
        assert len(values) == 100
        for args in values:
            assert isinstance(args, tuple) and len(args) == 1
            assert isinstance(args[0], tuple) and len(args[0]) == 2
            assert isinstance(args[0][0], str)
            assert isinstance(args[0][1], int)


def test_strings_contain_no_lone_surrogates() -> None:
    function = function_ref("def f(s: str) -> str:\n    return s\n")
    values = draws(strategies_source(function))
    assert len(values) == 100
    for args in values:
        assert isinstance(args[0], str)
        assert all(unicodedata.category(char) != "Cs" for char in args[0])


def test_bool_draws_booleans() -> None:
    function = function_ref("def f(flag: bool) -> bool:\n    return flag\n")
    values = draws(strategies_source(function), count=10)
    assert values
    assert {args[0] for args in values} == {True, False}


def test_parameter_without_hint_raises_no_types() -> None:
    function = function_ref("def f(x):\n    return x\n")
    with pytest.raises(UnsupportedConstruct) as excinfo:
        strategies_source(function)
    assert excinfo.value.construct == "no types"


def test_unsupported_hint_raises_type() -> None:
    function = function_ref("def f(x: dict[str, int]) -> int:\n    return 0\n")
    with pytest.raises(UnsupportedConstruct) as excinfo:
        strategies_source(function)
    assert excinfo.value.construct == "type dict[str, int]"


def test_unsupported_bare_hint_raises_type() -> None:
    function = function_ref("def f(x: set) -> int:\n    return 0\n")
    with pytest.raises(UnsupportedConstruct) as excinfo:
        strategies_source(function)
    assert excinfo.value.construct == "type set"
