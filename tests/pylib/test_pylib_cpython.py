"""Every PyLib block, compiled to Python with Dafny, must behave exactly like CPython."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Literal

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pcc.config import Config
from pcc.pylib import PYLIB_FILE

pytestmark = pytest.mark.dafny

nonzero = st.integers().filter(lambda b: b != 0)

# Python strings hold no lone surrogates in this project (docs/plan.md), and Dafny has no
# surrogate code points either, so strings are drawn without the Cs category.
_SURROGATE_EXCLUSION: Final[tuple[Literal["Cs"]]] = ("Cs",)
no_surrogates = st.text(alphabet=st.characters(exclude_categories=_SURROGATE_EXCLUSION), max_size=30)
int_lists = st.lists(st.integers(), max_size=30)
sequences = st.one_of(no_surrogates, int_lists)


def to_dafny(xs: str | list[int]) -> Any:
    """Wrap a Python sequence as the Dafny runtime sequence the compiled blocks expect."""
    return importlib.import_module("_dafny").Seq(xs)


def from_dafny(result: Any) -> list[Any]:
    """Unwrap a Dafny runtime sequence into plain elements, so it compares with CPython."""
    return list(result)


# Modules imported by the Dafny-generated Python code. Removed after the fixture so that a later build in the same
# process does not reuse them.
_GENERATED_MODULES = ("PyLib", "_dafny", "System_", "module_")


@pytest.fixture(scope="module")
def pylib(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ModuleType]:
    out = tmp_path_factory.mktemp("pylib") / "out"
    subprocess.run(
        ["dafny", "build", "--target:py", "--no-verify", "-o", str(out), str(PYLIB_FILE)],
        check=True,
        capture_output=True,
    )
    module_dir = Path(f"{out}-py")
    sys.path.insert(0, str(module_dir))
    try:
        yield importlib.import_module("PyLib").default__
    finally:
        sys.path.remove(str(module_dir))
        for name in _GENERATED_MODULES:
            sys.modules.pop(name, None)


def test_verifies() -> None:
    limit = str(Config().dafny.resource_limit)
    proc = subprocess.run(
        ["dafny", "verify", "--resource-limit", limit, "--verification-time-limit", "0", str(PYLIB_FILE)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@settings(max_examples=2000)
@given(a=st.integers(), b=nonzero)
def test_floordiv_matches_cpython(pylib: ModuleType, a: int, b: int) -> None:
    assert pylib.py__floordiv(a, b) == a // b


@settings(max_examples=2000)
@given(a=st.integers(), b=nonzero)
def test_mod_matches_cpython(pylib: ModuleType, a: int, b: int) -> None:
    assert pylib.py__mod(a, b) == a % b


@pytest.mark.parametrize(("a", "b"), [(7, -2), (-7, 2), (-7, -2), (0, -3), (6, -3), (-10, -11)])
def test_known_differences_from_dafny_division(pylib: ModuleType, a: int, b: int) -> None:
    assert (pylib.py__floordiv(a, b), pylib.py__mod(a, b)) == (a // b, a % b)


@settings(max_examples=500)
@given(xs=sequences)
def test_len_matches_cpython(pylib: ModuleType, xs: str | list[int]) -> None:
    assert pylib.py__len(xs) == len(xs)


@settings(max_examples=500)
@given(xs=sequences, offset=st.integers())
def test_index_matches_cpython(pylib: ModuleType, xs: str | list[int], offset: int) -> None:
    assume(len(xs) > 0)
    i = offset % (2 * len(xs)) - len(xs)
    assert pylib.py__index(xs, i) == xs[i]


@settings(max_examples=500)
@given(xs=sequences, start=st.integers(), stop=st.integers())
def test_slice_matches_cpython(pylib: ModuleType, xs: str | list[int], start: int, stop: int) -> None:
    assert from_dafny(pylib.py__slice(to_dafny(xs), start, stop)) == list(xs[start:stop])


@settings(max_examples=500)
@given(xs=sequences)
def test_reversed_matches_cpython(pylib: ModuleType, xs: str | list[int]) -> None:
    assert from_dafny(pylib.py__reversed(to_dafny(xs))) == list(xs[::-1])


@pytest.mark.parametrize(
    ("xs", "start", "stop", "expected"),
    [
        ("abcdba", 1, -1, "bcdb"),
        ("ab", 5, 10, ""),
        ("abc", -10, 2, "ab"),
        ("", 0, 5, ""),
        ("abc", 1, 1, ""),
        ("abc", 2, 0, ""),
        ("abcde", -4, -2, "bc"),
        ("abc", -100, 100, "abc"),
    ],
)
def test_slice_edge_cases(pylib: ModuleType, xs: str, start: int, stop: int, expected: str) -> None:
    assert from_dafny(pylib.py__slice(to_dafny(xs), start, stop)) == list(expected) == list(xs[start:stop])


@pytest.mark.parametrize(
    ("xs", "i", "expected"),
    [
        ("abc", -1, "c"),
        ("abc", -3, "a"),
        ("abc", 0, "a"),
        ("abc", 2, "c"),
        ([1, 2, 3], -1, 3),
        ([1, 2, 3], 0, 1),
    ],
)
def test_index_edge_cases(pylib: ModuleType, xs: str | list[int], i: int, expected: str | int) -> None:
    assert pylib.py__index(xs, i) == expected == xs[i]


@pytest.mark.parametrize(
    ("xs", "expected"),
    [
        ("", ""),
        ("a", "a"),
        ("abc", "cba"),
        ([], []),
        ([1], [1]),
        ([1, 2, 3], [3, 2, 1]),
    ],
)
def test_reversed_edge_cases(pylib: ModuleType, xs: str | list[int], expected: str | list[int]) -> None:
    assert from_dafny(pylib.py__reversed(to_dafny(xs))) == list(expected) == list(xs[::-1])


@settings(max_examples=2000)
@given(a=st.integers(), b=st.integers())
def test_min_matches_cpython(pylib: ModuleType, a: int, b: int) -> None:
    assert pylib.py__min(a, b) == min(a, b)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(3, 7, 3), (7, 3, 3), (5, 5, 5), (-2, -8, -8), (0, -1, -1), (-5, 4, -5)],
)
def test_min_edge_cases(pylib: ModuleType, a: int, b: int, expected: int) -> None:
    assert pylib.py__min(a, b) == expected == min(a, b)


@settings(max_examples=500)
@given(xs=int_lists, n=st.integers())
def test_count_matches_cpython(pylib: ModuleType, xs: list[int], n: int) -> None:
    assert pylib.py__count(to_dafny(xs), lambda v: v >= n) == sum(1 for v in xs if v >= n)


@settings(max_examples=500)
@given(xs=no_surrogates)
def test_count_chars_matches_cpython(pylib: ModuleType, xs: str) -> None:
    assert pylib.py__count(to_dafny(xs), lambda v: v == "a") == sum(1 for v in xs if v == "a")


@pytest.mark.parametrize(
    ("xs", "predicate", "expected"),
    [
        ([], lambda v: v >= 0, 0),
        ([1, 2, 3], lambda v: v >= 5, 0),
        ([1, 2, 3], lambda v: v >= 0, 3),
        ([1, -1, 2, -2], lambda v: v < 0, 2),
        ("abca", lambda v: v == "a", 2),
    ],
)
def test_count_edge_cases(pylib: ModuleType, xs: str | list[int], predicate: Any, expected: int) -> None:
    assert pylib.py__count(to_dafny(xs), predicate) == expected


@settings(max_examples=500)
@given(xs=int_lists, offset=st.integers(), v=st.integers())
def test_set_index_matches_cpython(pylib: ModuleType, xs: list[int], offset: int, v: int) -> None:
    assume(len(xs) > 0)
    i = offset % (2 * len(xs)) - len(xs)
    expected = list(xs)
    expected[i] = v
    assert from_dafny(pylib.py__set__index(to_dafny(xs), i, v)) == expected


@pytest.mark.parametrize(
    ("xs", "i", "v", "expected"),
    [
        ([1, 2, 3], 0, 9, [9, 2, 3]),
        ([1, 2, 3], -1, 9, [1, 2, 9]),
        ([1, 2, 3], -3, 9, [9, 2, 3]),
        ([1], 0, 9, [9]),
        ([1], -1, 9, [9]),
        ([1, 2, 3], 2, 7, [1, 2, 7]),
    ],
)
def test_set_index_edge_cases(pylib: ModuleType, xs: list[int], i: int, v: int, expected: list[int]) -> None:
    result = list(xs)
    result[i] = v
    assert from_dafny(pylib.py__set__index(to_dafny(xs), i, v)) == expected == result
