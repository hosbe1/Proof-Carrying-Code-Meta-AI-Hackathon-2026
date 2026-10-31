"""Tests for ``pcc.verifier.ghost``: the proof may only change ghost code."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from pcc.verifier.anti_cheat import _normalize
from pcc.verifier.ghost import (
    declared_lemma_names,
    executable_part,
    same_executable,
    spec_referenced_lemmas,
    strip_ghost,
)


def test_example_loop_invariant_and_decreases_are_ghost() -> None:
    original = "method M(s: seq<int>) returns (r: int)\n{\n  var i := 0;\n  while i < |s| {\n    i := i + 1;\n  }\n}\n"
    proven = (
        "method M(s: seq<int>) returns (r: int)\n"
        "{\n  var i := 0;\n  while i < |s|\n"
        "    invariant 0 <= i <= |s|\n"
        "    decreases |s| - i\n"
        "  {\n    i := i + 1;\n  }\n}\n"
    )
    assert same_executable(original, proven, lemma_names=[]) is True


def test_example_body_change_is_not_ghost() -> None:
    original = "method M() returns (r: int)\n{\n  var x := 0;\n  if x <= 2 {\n    x := x + 1;\n  }\n}\n"
    proven = "method M() returns (r: int)\n{\n  var x := 0;\n  if x <= 1 {\n    x := x + 1;\n  }\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is False


def test_example_lemma_call_in_list_is_ghost() -> None:
    original = "method M(s: seq<int>)\n{\n  var i := 0;\n  i := i + 1;\n}\n"
    proven = "method M(s: seq<int>)\n{\n  var i := 0;\n  LemmaReverse(s);\n  i := i + 1;\n}\n"
    assert same_executable(original, proven, lemma_names=["LemmaReverse"]) is True


def test_example_call_missing_from_list_is_not_ghost() -> None:
    original = "method M(s: seq<int>)\n{\n  var i := 0;\n  i := i + 1;\n}\n"
    proven = "method M(s: seq<int>)\n{\n  var i := 0;\n  LemmaReverse(s);\n  i := i + 1;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is False


def test_edge_assert_inside_nested_block_is_ghost() -> None:
    original = "method M(x: int)\n{\n  if x > 0 {\n    if x > 1 {\n      var y := x;\n    }\n  }\n}\n"
    proven = (
        "method M(x: int)\n{\n  if x > 0 {\n    if x > 1 {\n      assert x > 1;\n      var y := x;\n    }\n  }\n}\n"
    )
    assert same_executable(original, proven, lemma_names=[]) is True


def test_edge_calc_with_nested_braces_is_ghost() -> None:
    original = "method M(x: int)\n{\n  var y := x;\n}\n"
    proven = "method M(x: int)\n{\n  calc {\n    x;\n    { x + 0; }\n    x;\n  }\n  var y := x;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_edge_decreases_on_method_header_is_ghost() -> None:
    original = "method M(n: int) returns (r: int)\n{\n  var x := n;\n}\n"
    proven = "method M(n: int) returns (r: int)\n  decreases n\n{\n  var x := n;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_modifies_clause_is_ghost() -> None:
    original = "method M() {\n  var x := 0;\n}\n"
    proven = "method M() modifies this {\n  var x := 0;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_reveal_is_ghost() -> None:
    original = "method M() {\n  var x := 0;\n}\n"
    proven = "method M() {\n  reveal Foo();\n  var x := 0;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_ghost_var_is_ghost() -> None:
    original = "method M() {\n  var x := 0;\n}\n"
    proven = "method M() {\n  ghost var g := 0;\n  var x := 0;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_assert_by_block_is_ghost() -> None:
    original = "method M(x: int) {\n  var y := x;\n}\n"
    proven = "method M(x: int) {\n  assert x == x by { calc { x; x; } }\n  var y := x;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_assert_by_lemma_call_is_ghost() -> None:
    original = "method M(x: int) {\n  var y := x;\n}\n"
    proven = "method M(x: int) {\n  assert x == x by LemmaHelper(x);\n  var y := x;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_qualified_lemma_call_is_not_ghost() -> None:
    original = "method M(s: seq<int>) {\n  var i := 0;\n}\n"
    proven = "method M(s: seq<int>) {\n  M.LemmaHelper(s);\n  var i := 0;\n}\n"
    assert same_executable(original, proven, lemma_names=["LemmaHelper"]) is False


def test_assigned_lemma_call_is_not_ghost() -> None:
    original = "method M(s: seq<int>) {\n  var i := 0;\n}\n"
    proven = "method M(s: seq<int>) {\n  var i := LemmaHelper(s);\n}\n"
    assert same_executable(original, proven, lemma_names=["LemmaHelper"]) is False


def test_ghost_prefixed_lemma_call_is_not_ghost() -> None:
    original = "method M(s: seq<int>) {\n  var i := 0;\n}\n"
    proven = "method M(s: seq<int>) {\n  ghost LemmaHelper(s);\n  var i := 0;\n}\n"
    assert same_executable(original, proven, lemma_names=["LemmaHelper"]) is False


def test_added_requires_is_not_ghost() -> None:
    original = "method M(x: int) {\n  var y := x;\n}\n"
    proven = "method M(x: int) requires x > 0 {\n  var y := x;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is False


def test_added_ensures_is_not_ghost() -> None:
    original = "method M(x: int) {\n  var y := x;\n}\n"
    proven = "method M(x: int) ensures y == x {\n  var y := x;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is False


def test_added_reads_is_not_ghost() -> None:
    original = "method M(x: int) {\n  var y := x;\n}\n"
    proven = "method M(x: int) reads this {\n  var y := x;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is False


def test_added_ghost_const_is_not_ghost() -> None:
    original = "method M(x: int) {\n  var y := x;\n}\n"
    proven = "method M(x: int) {\n  ghost const g := 1;\n  var y := x;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is False


def test_keyword_inside_string_is_kept() -> None:
    original = 'method M() {\n  var s := "hello";\n}\n'
    proven = 'method M() {\n  var s := "calc invariant";\n}\n'
    assert same_executable(original, proven, lemma_names=[]) is False


def test_keyword_inside_line_comment_is_ghost() -> None:
    original = "method M() {\n  var x := 0;\n}\n"
    proven = "method M() {\n  // invariant 0 <= x\n  var x := 0;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_keyword_inside_block_comment_is_ghost() -> None:
    original = "method M() {\n  var x := 0;\n}\n"
    proven = "method M() {\n  /* invariant 0 <= x */\n  var x := 0;\n}\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_whitespace_only_difference_is_ghost() -> None:
    original = "method M() {\n\tvar x := 0;\n}\n"
    proven = "method   M()   {  var   x   :=   0;   }\n"
    assert same_executable(original, proven, lemma_names=[]) is True


def test_unbalanced_braces_give_false() -> None:
    original = "method M() {\n  var x := 0;\n}\n"
    proven = "method M() {\n  var x := 0;\n"
    assert same_executable(original, proven, lemma_names=[]) is False
    assert same_executable(proven, original, lemma_names=[]) is False


def test_executable_part_never_raises_on_unbalanced() -> None:
    unbalanced = "method M() {\n  calc {\n    x;\n"
    assert isinstance(executable_part(unbalanced, lemma_names=[]), str)


def test_lexer_is_reused_not_copied() -> None:
    source = Path(__file__).resolve().parents[2] / "src" / "pcc" / "verifier" / "ghost.py"
    text = source.read_text(encoding="utf-8")
    assert "anti_cheat" in text
    assert 'startswith("//"' not in text
    assert 'startswith("/*"' not in text


SUM_ANSWER = (
    "method Sum(n: int) returns (r: int)\n"
    "  requires n >= 0\n"
    "  ensures r >= 0\n"
    "{\n"
    "  r := 0;\n"
    "  var i := 0;\n"
    "  while i < n\n"
    "    invariant 0 <= i <= n\n"
    "    decreases n - i\n"
    "  {\n"
    "    r := r + i;\n"
    "    LemmaAddNonneg(r, i);\n"
    "    i := i + 1;\n"
    "  }\n"
    "}\n"
    "\n"
    "lemma LemmaAddNonneg(a: int, b: int)\n"
    "  requires a >= 0 && b >= 0\n"
    "  ensures a + b >= 0\n"
    "{\n"
    "}\n"
)

PLAIN_ANSWER = "method Double(x: int) returns (r: int)\n  ensures r == 2 * x\n{\n  r := x + x;\n}\n"


def test_example_strip_ghost_removes_invariant_decreases_call_and_lemma() -> None:
    program = strip_ghost(SUM_ANSWER, lemma_names=["LemmaAddNonneg"])
    assert "invariant" not in program
    assert "LemmaAddNonneg" not in program
    assert "requires n >= 0" in program
    assert "ensures r >= 0" in program
    assert "r := r + i;" in program
    assert "i := i + 1;" in program


def test_strip_ghost_keeps_character_positions() -> None:
    program = strip_ghost(SUM_ANSWER, lemma_names=["LemmaAddNonneg"])
    assert len(program) == len(SUM_ANSWER)
    assert [len(line) for line in program.split("\n")] == [len(line) for line in SUM_ANSWER.split("\n")]
    for kept, original in zip(program.split("\n"), SUM_ANSWER.split("\n"), strict=True):
        stripped = kept.strip()
        if stripped:
            assert original.startswith(kept[: len(kept) - len(kept.lstrip())])


def test_answer_without_ghost_is_unchanged_after_normalize() -> None:
    program = strip_ghost(PLAIN_ANSWER, lemma_names=[])
    assert _normalize(program) == _normalize(PLAIN_ANSWER)


def test_edge_nested_braces_in_lemma_body_are_stripped_together() -> None:
    answer = (
        "method M(x: int) returns (r: int)\n"
        "{\n"
        "  if x > 0 {\n"
        "    r := Helper(x);\n"
        "  } else {\n"
        "    r := 0;\n"
        "  }\n"
        "}\n"
        "\n"
        "lemma HelperLemma(x: int) returns (r: int)\n"
        "  ensures r >= 0\n"
        "{\n"
        "  if x > 0 {\n"
        "    calc {\n"
        "      x;\n"
        "      { x + 0; }\n"
        "      x;\n"
        "    }\n"
        "    r := x;\n"
        "  } else {\n"
        "    r := 0;\n"
        "  }\n"
        "}\n"
    )
    assert declared_lemma_names(answer) == ("HelperLemma",)
    program = strip_ghost(answer, lemma_names=["HelperLemma"])
    assert "HelperLemma" not in program
    assert "calc" not in program
    assert "r := Helper(x);" in program
    assert "r := 0;" in program


def test_edge_decreases_on_method_header_is_kept() -> None:
    answer = "method M(n: int) returns (r: int)\n  decreases n\n{\n  var x := n;\n}\n"
    program = strip_ghost(answer, lemma_names=[])
    assert "decreases n" in program
    assert "var x := n;" in program


def test_edge_lemma_used_in_spec_is_kept_with_warning() -> None:
    answer = (
        "method M(x: int) returns (r: int)\n"
        "  ensures LemmaProp(x, r)\n"
        "{\n"
        "  LemmaProp(x, 0);\n"
        "  r := 0;\n"
        "}\n"
        "\n"
        "lemma LemmaProp(x: int, r: int)\n"
        "  ensures r >= 0\n"
        "{\n"
        "}\n"
    )
    assert spec_referenced_lemmas(answer, ["LemmaProp"]) == ("LemmaProp",)
    with pytest.warns(UserWarning, match="LemmaProp"):
        program = strip_ghost(answer, lemma_names=["LemmaProp"])
    assert "lemma LemmaProp" in program
    assert "ensures r >= 0" in program
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _normalize(program) != _normalize(answer)


def test_lemma_modifiers_and_duplicates() -> None:
    answer = "ghost lemma G() {}\nstatic lemma S() {}\nopaque lemma O() {}\nlemma Plain() {}\nghost lemma G() {}\n"
    assert declared_lemma_names(answer) == ("G", "S", "O", "Plain")


def test_lemma_keyword_in_string_or_comment_is_no_declaration() -> None:
    answer = (
        "method M() {\n"
        '  var s := "lemma Fake() {";\n'
        "  // lemma Commented() {}\n"
        "  /* lemma Blocked() {} */\n"
        "  var t := 'c';\n"
        "}\n"
    )
    assert declared_lemma_names(answer) == ()
    assert spec_referenced_lemmas(answer, ["Fake", "Commented", "Blocked"]) == ()


def test_strip_ghost_never_raises_on_unbalanced() -> None:
    unbalanced = "lemma L(x: int) {\n  assert x == x;\n"
    assert isinstance(strip_ghost(unbalanced, lemma_names=["L"]), str)


def test_executable_part_drops_lemma_declarations() -> None:
    lemma = "lemma L(x: int)\n  ensures x >= 0\n{\n}\n"
    method = "method M(x: int) {\n  var y := x;\n}\n"
    assert executable_part(method + lemma, lemma_names=["L"]) == executable_part(method, lemma_names=["L"])


GHOST_FIXTURES = [
    SUM_ANSWER,
    PLAIN_ANSWER,
    "method M(x: int) {\n  var y := x;\n}\n",
    "method M(n: int) returns (r: int)\n  decreases n\n{\n  var x := n;\n}\n",
    (
        "method M(x: int) returns (r: int)\n"
        "  ensures LemmaProp(x, r)\n"
        "{\n"
        "  LemmaProp(x, 0);\n"
        "  r := 0;\n"
        "}\n"
        "\n"
        "lemma LemmaProp(x: int, r: int)\n"
        "  ensures r >= 0\n"
        "{\n"
        "}\n"
    ),
    (
        "method M() {\n"
        '  var s := "lemma Fake() { calc { x; } }";\n'
        "  // invariant 0 <= x\n"
        "  var x := 0;\n"
        "  assert x == x by { calc { x; x; } }\n"
        "}\n"
    ),
]


@pytest.mark.parametrize("answer", GHOST_FIXTURES)
def test_property_executable_part_of_strip_ghost_is_stable(answer: str) -> None:
    names = declared_lemma_names(answer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = strip_ghost(answer, lemma_names=names)
    assert executable_part(program, lemma_names=names) == executable_part(answer, lemma_names=names)
