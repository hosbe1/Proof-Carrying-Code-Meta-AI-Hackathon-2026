"""Round-trip tests for ``pcc.equivalence.convert`` against a freshly compiled Dafny runtime.

The ``dafny_runtime`` fixture builds a tiny Dafny module with ``dafny build`` and imports the ``_dafny`` runtime it
generates, so the converter is tested against the real runtime the back-compiled code uses, not a stub. Tests that
need the CLI carry the ``dafny`` marker and are skipped when it is missing; the import test runs everywhere.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import typing
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pcc.equivalence import convert

PROBE_DAFNY = """module ConvertProbe {
  method EchoInt(x: int) returns (r: int) { r := x; }
  method EchoBool(b: bool) returns (r: bool) { r := b; }
  method EchoStr(s: string) returns (r: string) { r := s; }
  method EchoInts(xs: seq<int>) returns (r: seq<int>) { r := xs; }
  method EchoPair(p: (string, int)) returns (r: (string, int)) { r := p; }
  method EchoNested(xss: seq<seq<int>>) returns (r: seq<seq<int>>) { r := xss; }
  method SampleStr() returns (s: string) { s := "abc"; }
  method SampleInts() returns (s: seq<int>) { s := [1, -2, 3]; }
  method SamplePair() returns (p: (string, int)) { p := ("a", 1); }
}
"""


class ProbeModules(NamedTuple):
    """The freshly built ``_dafny`` runtime next to the compiled probe module."""

    dafny: ModuleType
    probe: ModuleType


@pytest.fixture(scope="module")
def dafny_modules(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ProbeModules]:
    """Build the probe module and import its ``_dafny`` runtime."""
    build_dir = tmp_path_factory.mktemp("convert-probe")
    (build_dir / "Probe.dfy").write_text(PROBE_DAFNY, encoding="utf-8")
    proc = subprocess.run(
        ["dafny", "build", "--target:py", "Probe.dfy"],
        cwd=build_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=300.0,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out_dir = str(build_dir / "Probe-py")
    sys.path.insert(0, out_dir)
    try:
        # The generated file is named after the Dafny module (`ConvertProbe`), not the `.dfy` file.
        yield ProbeModules(dafny=importlib.import_module("_dafny"), probe=importlib.import_module("ConvertProbe"))
    finally:
        sys.path.remove(out_dir)
        sys.modules.pop("_dafny", None)
        sys.modules.pop("ConvertProbe", None)


def echo(modules: ProbeModules, name: str, value: object) -> object:
    """Call a ``ConvertProbe`` echo method on ``value`` through the compiled code."""
    namespace: Any = getattr(modules.probe, "default__")  # noqa: B009 -- module built by `dafny build`, unknown to mypy
    method: Any = getattr(namespace, name)
    return method(value)


def seq_type(modules: ProbeModules) -> Any:
    """The ``_dafny.Seq`` class of the freshly built runtime."""
    return modules.dafny.Seq


def test_convert_imports_only_stdlib() -> None:
    """``convert`` embeds into the sandbox wrapper, so it must not import ``pcc`` and needs ``_dafny`` lazily."""
    assert convert.__file__ is not None
    source = Path(convert.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level: set[str] = set()
    nested: set[str] = set()
    for node in tree.body:
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                names = {alias.name.split(".")[0] for alias in child.names}
                (top_level if child is node else nested).update(names)
            elif isinstance(child, ast.ImportFrom):
                assert child.level == 0, "relative imports are not allowed"
                assert child.module is not None
                name = child.module.split(".")[0]
                (top_level if child is node else nested).add(name)
    assert "pcc" not in top_level and "pcc" not in nested
    assert top_level <= sys.stdlib_module_names, top_level - sys.stdlib_module_names
    assert nested <= sys.stdlib_module_names | {"_dafny"}, nested - sys.stdlib_module_names - {"_dafny"}


@pytest.mark.dafny
@given(value=st.integers())
def test_round_trip_int(dafny_modules: ProbeModules, value: int) -> None:
    converted = convert.to_dafny(value, int)
    assert type(converted) is int
    assert convert.to_python(converted, int) == value


@pytest.mark.dafny
@given(value=st.booleans())
def test_round_trip_bool(dafny_modules: ProbeModules, value: bool) -> None:
    converted = convert.to_dafny(value, bool)
    assert type(converted) is bool
    assert convert.to_python(converted, bool) == value


@pytest.mark.dafny
@given(value=st.text())
def test_round_trip_str(dafny_modules: ProbeModules, value: str) -> None:
    converted = convert.to_dafny(value, str)
    assert isinstance(converted, seq_type(dafny_modules))
    assert convert.to_python(converted, str) == value


@pytest.mark.dafny
@given(value=st.lists(st.integers()))
def test_round_trip_list_int(dafny_modules: ProbeModules, value: list[int]) -> None:
    converted = convert.to_dafny(value, list[int])
    assert isinstance(converted, seq_type(dafny_modules))
    assert convert.to_python(converted, list[int]) == value


@pytest.mark.dafny
@given(value=st.lists(st.integers()))
def test_round_trip_typing_list_int(dafny_modules: ProbeModules, value: list[int]) -> None:
    converted = convert.to_dafny(value, typing.List[int])  # noqa: UP006 -- the ticket requires typing.List support
    assert isinstance(converted, seq_type(dafny_modules))
    assert convert.to_python(converted, typing.List[int]) == value  # noqa: UP006 -- the ticket requires typing.List support


@pytest.mark.dafny
@given(value=st.lists(st.lists(st.integers())))
def test_round_trip_nested_list_int(dafny_modules: ProbeModules, value: list[list[int]]) -> None:
    converted = convert.to_dafny(value, list[list[int]])
    assert convert.to_python(converted, list[list[int]]) == value


@pytest.mark.dafny
@given(value=st.tuples(st.text(), st.integers()))
def test_round_trip_tuple_str_int(dafny_modules: ProbeModules, value: tuple[str, int]) -> None:
    converted = convert.to_dafny(value, tuple[str, int])
    assert type(converted) is tuple
    assert convert.to_python(converted, tuple[str, int]) == value


@pytest.mark.dafny
def test_ticket_examples(dafny_modules: ProbeModules) -> None:
    """The examples and edge cases from the ticket, through the compiled probe code."""
    cases: list[tuple[object, object, str]] = [
        ("abc", str, "EchoStr"),
        ([1, -2, 3], list[int], "EchoInts"),
        (("a", 1), tuple[str, int], "EchoPair"),
        ("", str, "EchoStr"),
        ([], list[int], "EchoInts"),
        ([[], [1, -2]], list[list[int]], "EchoNested"),
        ("ß", str, "EchoStr"),
        ("aß🎉z", str, "EchoStr"),
    ]
    for value, py_type, method in cases:
        assert convert.to_python(convert.to_dafny(value, py_type), py_type) == value
        assert convert.to_python(echo(dafny_modules, method, convert.to_dafny(value, py_type)), py_type) == value


@pytest.mark.dafny
def test_compiled_samples_convert_back(dafny_modules: ProbeModules) -> None:
    """Values produced by real compiled Dafny code convert back to the expected Python values."""
    namespace: Any = getattr(dafny_modules.probe, "default__")  # noqa: B009 -- module built by `dafny build`, unknown to mypy
    assert convert.to_python(namespace.SampleStr(), str) == "abc"
    assert convert.to_python(namespace.SampleInts(), list[int]) == [1, -2, 3]
    assert convert.to_python(namespace.SamplePair(), tuple[str, int]) == ("a", 1)


@pytest.mark.dafny
@given(value=st.integers())
def test_echo_int_through_compiled_code(dafny_modules: ProbeModules, value: int) -> None:
    assert convert.to_python(echo(dafny_modules, "EchoInt", convert.to_dafny(value, int)), int) == value


@pytest.mark.dafny
@given(value=st.booleans())
def test_echo_bool_through_compiled_code(dafny_modules: ProbeModules, value: bool) -> None:
    assert convert.to_python(echo(dafny_modules, "EchoBool", convert.to_dafny(value, bool)), bool) == value


@pytest.mark.dafny
@given(value=st.text())
def test_echo_str_through_compiled_code(dafny_modules: ProbeModules, value: str) -> None:
    assert convert.to_python(echo(dafny_modules, "EchoStr", convert.to_dafny(value, str)), str) == value


@pytest.mark.dafny
@given(value=st.lists(st.integers()))
def test_echo_ints_through_compiled_code(dafny_modules: ProbeModules, value: list[int]) -> None:
    assert convert.to_python(echo(dafny_modules, "EchoInts", convert.to_dafny(value, list[int])), list[int]) == value


@pytest.mark.dafny
@given(value=st.tuples(st.text(), st.integers()))
def test_echo_pair_through_compiled_code(dafny_modules: ProbeModules, value: tuple[str, int]) -> None:
    assert (
        convert.to_python(echo(dafny_modules, "EchoPair", convert.to_dafny(value, tuple[str, int])), tuple[str, int])
        == value
    )


@pytest.mark.dafny
@given(value=st.lists(st.lists(st.integers())))
def test_echo_nested_through_compiled_code(dafny_modules: ProbeModules, value: list[list[int]]) -> None:
    assert (
        convert.to_python(echo(dafny_modules, "EchoNested", convert.to_dafny(value, list[list[int]])), list[list[int]])
        == value
    )


@pytest.mark.dafny
def test_unsupported_types_raise(dafny_modules: ProbeModules) -> None:
    """``dict``, ``set`` and ``float`` are out of scope; bare ``list``/``tuple`` carry no element types."""
    unsupported: list[object] = [dict[str, int], set[int], float, list, tuple, None, int | str]
    for py_type in unsupported:
        with pytest.raises(TypeError):
            convert.to_dafny(0, py_type)
        with pytest.raises(TypeError):
            convert.to_python(0, py_type)


@pytest.mark.dafny
def test_mismatched_values_raise(dafny_modules: ProbeModules) -> None:
    """A value of the wrong shape for its annotation raises ``TypeError`` instead of converting silently."""
    seq: Any = seq_type(dafny_modules)
    mismatched: list[tuple[object, object]] = [
        ("x", int),
        (1, str),
        (True, int),
        (1, bool),
        ((1, 2), list[int]),
        ("ab", list[int]),
        (seq([1]), str),
        ([1], tuple[int, int]),
        (("a", 1, 2), tuple[str, int]),
        (("a", "b"), tuple[str, int]),
    ]
    for value, py_type in mismatched:
        with pytest.raises(TypeError):
            convert.to_dafny(value, py_type)
        with pytest.raises(TypeError):
            convert.to_python(value, py_type)
