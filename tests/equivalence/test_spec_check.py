"""The transpiled spec agrees with the Python spec on the pipeline's cases (docs/plan.md, step 4).

The Dafny `pre`/`spec` predicates are compiled back to Python and evaluated next to the Python `pre`/`spec` on
every case. Any disagreement fails the check with the first differing case in the detail.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Final

import pytest

from pcc.core import CheckKind, CheckRecord, FunctionRef, NotCheckable, Spec, ToolError, source_hash
from pcc.equivalence import spec_check as spec_check_module
from pcc.equivalence.spec_check import check_spec

FIXTURE_DIR: Final = Path(__file__).with_name("fixtures")

PALINDROME_SOURCE: Final = "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n"
PALINDROME_PREDICATE: Final = "def spec(s, result):\n    return result == (s == s[::-1])\n"
PALINDROME_PRECONDITION: Final = "def pre(s):\n    return True\n"

FIRST_SOURCE: Final = "def first(xs: list[int]) -> int:\n    return xs[0]\n"
FIRST_PRECONDITION: Final = "def pre(xs):\n    return len(xs) > 0\n"
FIRST_PREDICATE: Final = "def spec(xs, result):\n    return result == xs[0]\n"
FIRST_PRE_TRUE: Final = "predicate pre(xs: seq<int>)\n{\n  true\n}\n"
FIRST_SPEC: Final = "predicate spec(xs: seq<int>, result: int)\n  requires pre(xs)\n{\n  result == xs[0]\n}\n"

DIV_SOURCE: Final = "def div10(x: int) -> int:\n    return 10 // x\n"
DIV_PREDICATE: Final = "def spec(x, result):\n    return result == 10 // x\n"
DIV_SPEC_TRUE: Final = "predicate spec(x: int, result: int)\n{\n  true\n}\n"

WRONG_SPEC_TRUE: Final = "predicate spec(s: string, result: bool)\n  requires pre(s)\n{\n  true\n}\n"


def function_ref(source: str, path: str, qualname: str) -> FunctionRef:
    """A ``FunctionRef`` with the hash computed from ``source``."""
    return FunctionRef(path, qualname, 1, source, source_hash(source))


def palindrome_spec() -> Spec:
    """The Python palindrome spec with a trivially true precondition."""
    function = function_ref(PALINDROME_SOURCE, "src/strings/palindrome.py", "is_palindrome")
    return Spec(function, PALINDROME_PREDICATE, PALINDROME_PRECONDITION)


def palindrome_cases(count: int = 200) -> tuple[tuple[str, str], ...]:
    """Deterministic palindrome cases: ``(repr(s), repr(s == s[::-1]))`` pairs with fixed edges first."""
    edges = ["", "a", "aa", "ab", "aba", "abc", "abba", "abcba", "abcdba", "abccaa"]
    rng = random.Random(38)
    drawn = ["".join(rng.choice("abc") for _ in range(rng.randint(0, 8))) for _ in range(count - len(edges))]
    return tuple((repr(s), repr(s == s[::-1])) for s in [*edges, *drawn])


def test_no_cases_needs_no_dafny(tmp_path: Path) -> None:
    """No cases: the check fails without touching Dafny."""
    spec = palindrome_spec()
    record = check_spec(spec, "predicate spec(s: string, result: bool)", spec.function, tmp_path, cases=[])
    assert record.kind is CheckKind.DIFFERENTIAL
    assert record.passed is False
    assert record.detail.startswith("spec:")
    assert "no cases" in record.detail


def test_build_python_not_checkable_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``NotCheckable`` from ``build_python`` escapes instead of becoming a record."""

    def fail(_dafny_code: str, _out_dir: Path) -> Path:
        raise NotCheckable("spec translation differs")

    monkeypatch.setattr(spec_check_module, "build_python", fail)
    with pytest.raises(NotCheckable):
        check_spec(
            palindrome_spec(),
            "predicate spec(s: string, result: bool)",
            palindrome_spec().function,
            tmp_path,
            cases=[("'aba'", "True")],
        )


def test_tool_error_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``ToolError`` from the sandbox escapes instead of becoming a record."""

    def fake_build(_dafny_code: str, _out_dir: Path) -> Path:
        return tmp_path

    def fail(_script: str, *, cwd: Path, **_kwargs: object) -> object:
        raise ToolError("boom")

    monkeypatch.setattr(spec_check_module, "build_python", fake_build)
    monkeypatch.setattr(spec_check_module, "run_python", fail)
    with pytest.raises(ToolError):
        check_spec(
            palindrome_spec(),
            "predicate spec(s: string, result: bool)",
            palindrome_spec().function,
            tmp_path,
            cases=[("'aba'", "True")],
        )


@pytest.mark.dafny
def test_palindrome_spec_agrees(tmp_path: Path) -> None:
    """The fixture predicates agree with the Python palindrome spec on 200 cases."""
    dafny_spec = (FIXTURE_DIR / "palindrome_spec.dfy").read_text(encoding="utf-8")
    spec = palindrome_spec()
    record = check_spec(spec, dafny_spec, spec.function, tmp_path, cases=palindrome_cases())
    assert record.kind is CheckKind.DIFFERENTIAL
    assert isinstance(record, CheckRecord)
    assert record.passed is True
    assert record.detail.startswith("spec:")


@pytest.mark.dafny
def test_planted_wrong_spec_is_caught(tmp_path: Path) -> None:
    """A spec translated as ``true`` is caught on a case with the wrong result."""
    dafny_pre = (FIXTURE_DIR / "palindrome_spec.dfy").read_text(encoding="utf-8").split("predicate spec")[0]
    spec = palindrome_spec()
    record = check_spec(spec, dafny_pre + WRONG_SPEC_TRUE, spec.function, tmp_path, cases=[("'ab'", "True")])
    assert record.kind is CheckKind.DIFFERENTIAL
    assert record.passed is False
    assert record.detail.startswith("spec:")
    assert "'ab'" in record.detail


@pytest.mark.dafny
def test_planted_wrong_pre_is_caught(tmp_path: Path) -> None:
    """A ``pre`` translated as ``true`` is caught on ``xs=[]`` while the Python ``pre`` rejects it."""
    function = function_ref(FIRST_SOURCE, "src/lists/first.py", "first")
    spec = Spec(function, FIRST_PREDICATE, FIRST_PRECONDITION)
    record = check_spec(
        spec,
        FIRST_PRE_TRUE + FIRST_SPEC,
        function,
        tmp_path,
        cases=[("[1]", "1"), ("[]", "0")],
    )
    assert record.kind is CheckKind.DIFFERENTIAL
    assert record.passed is False
    assert record.detail.startswith("spec:")
    assert "[]" in record.detail


@pytest.mark.dafny
def test_python_spec_error_counts_as_false(tmp_path: Path) -> None:
    """A case where the Python ``spec`` raises counts as ``False`` and disagrees with a Dafny ``True``."""
    function = function_ref(DIV_SOURCE, "src/numbers/div.py", "div10")
    spec = Spec(function, DIV_PREDICATE, None)
    record = check_spec(spec, DIV_SPEC_TRUE, function, tmp_path, cases=[("2", "5"), ("0", "0")])
    assert record.kind is CheckKind.DIFFERENTIAL
    assert record.passed is False
    assert record.detail.startswith("spec:")
    assert "0" in record.detail
