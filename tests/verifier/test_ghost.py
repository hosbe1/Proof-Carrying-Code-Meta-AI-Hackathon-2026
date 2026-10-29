"""Tests for ``pcc.verifier.ghost``: the proof may only change ghost code."""

from __future__ import annotations

from pathlib import Path

from pcc.verifier.ghost import executable_part, same_executable


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
