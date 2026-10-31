"""Tests for ``pcc.transpile.types``: the Python to Dafny type mapping."""

from __future__ import annotations

import ast

import pytest

from pcc.core import UnsupportedConstruct
from pcc.transpile.types import dafny_type


def annotation_of(source: str) -> ast.expr | None:
    """The annotation of the single parameter of the function in ``source``."""
    tree = ast.parse(source)
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    return func.args.args[0].annotation


def map_param(source: str, *, lineno: int = 1) -> str:
    """Map the single parameter annotation of ``source`` to Dafny."""
    node = annotation_of(source)
    assert node is not None
    return dafny_type(node, lineno=lineno)


def test_int_bool_str() -> None:
    assert map_param("def f(a: int) -> int:\n    return a\n") == "int"
    assert map_param("def f(a: bool) -> bool:\n    return a\n") == "bool"
    assert map_param("def f(a: str) -> str:\n    return a\n") == "string"


def test_list_and_typing_list() -> None:
    assert map_param("def f(a: list[int]) -> int:\n    return 0\n") == "seq<int>"
    assert map_param("def f(a: list[str]) -> str:\n    return a\n") == "seq<string>"
    assert map_param("def f(a: list[bool]) -> bool:\n    return a\n") == "seq<bool>"
    assert map_param("def f(a: list[list[int]]) -> int:\n    return 0\n") == "seq<seq<int>>"


def test_typing_list_needs_typing_prefix() -> None:
    assert map_param("def f(a: typing.List[int]) -> int:\n    return 0\n") == "seq<int>"


def test_bare_list_name_without_parameter() -> None:
    with pytest.raises(UnsupportedConstruct, match="type list"):
        map_param("def f(a: list) -> int:\n    return 0\n")


def test_tuple_raises_type_tuple() -> None:
    with pytest.raises(UnsupportedConstruct) as exc_info:
        map_param("def f(a: tuple[int, str]) -> int:\n    return 0\n")
    assert exc_info.value.construct == "type tuple"
    with pytest.raises(UnsupportedConstruct) as exc_info:
        map_param("def f(a: typing.Tuple[int]) -> int:\n    return 0\n")
    assert exc_info.value.construct == "type tuple"
    with pytest.raises(UnsupportedConstruct) as exc_info:
        map_param("def f(a: tuple) -> int:\n    return 0\n")
    assert exc_info.value.construct == "type tuple"


def test_unknown_type_names_the_annotation() -> None:
    with pytest.raises(UnsupportedConstruct) as exc_info:
        map_param("def f(a: float) -> int:\n    return 0\n")
    assert exc_info.value.construct == "type float"
    with pytest.raises(UnsupportedConstruct) as exc_info:
        map_param("def f(a: dict[str, int]) -> int:\n    return 0\n")
    assert exc_info.value.construct == "type dict[str, int]"


def test_missing_annotation_raises_no_types() -> None:
    with pytest.raises(UnsupportedConstruct) as exc_info:
        dafny_type(None, lineno=2)
    assert exc_info.value.construct == "no types"
    assert exc_info.value.lineno == 2
