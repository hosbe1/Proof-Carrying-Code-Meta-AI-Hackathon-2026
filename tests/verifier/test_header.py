"""Tests for the Vericoding method-header check (issue #74).

Pure string tests: no Dafny CLI is needed. The examples and edge cases come straight
from the ticket; the duplicate, body-less and unbalanced cases pin the follow-up
decisions on the issue.
"""

from __future__ import annotations

import importlib

from pcc.verifier.header import header_violations

# ``pcc.verifier.anti_cheat`` resolves to the re-exported function, not the submodule,
# so go through the module cache instead.
anti_cheat_module = importlib.import_module("pcc.verifier.anti_cheat")
header_module = importlib.import_module("pcc.verifier.header")

SPEC = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n {}"

SAME = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n { r := n; }"


def test_same_header_with_body_gives_no_violation() -> None:
    assert header_violations(SAME, SPEC) == ()


def test_added_requires_is_changed() -> None:
    answer = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n requires false { r := n; }"
    assert header_violations(answer, SPEC) == ("header changed: M",)


def test_added_decreases_is_changed() -> None:
    answer = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n decreases n { r := n; }"
    assert header_violations(answer, SPEC) == ("header changed: M",)


def test_multiline_header_with_comment_gives_no_violation() -> None:
    answer = (
        "method M(n: int)\n"
        "  returns (r: int)\n"
        "  // the precondition\n"
        "  requires n >= 0 ensures r == n\n"
        "{\n"
        "  r := n;\n"
        "}\n"
    )
    assert header_violations(answer, SPEC) == ()


def test_renamed_method_is_missing() -> None:
    answer = SAME.replace("method M(", "method M2(")
    assert header_violations(answer, SPEC) == ("header missing: M",)


def test_two_declarations_second_changed() -> None:
    spec = (
        "function Add(a: int, b: int): int { a + b }\n"
        "method M(n: int) returns (r: int) requires n >= 0 ensures r == n {}"
    )
    answer = (
        "function Add(a: int, b: int): int { a + b }\n"
        "method M(n: int) returns (r: int) requires n >= 0 ensures r == n requires false { r := n; }"
    )
    assert header_violations(answer, spec) == ("header changed: M",)


def test_two_declarations_first_changed() -> None:
    spec = (
        "function Add(a: int, b: int): int { a + b }\n"
        "method M(n: int) returns (r: int) requires n >= 0 ensures r == n {}"
    )
    answer = (
        "function Add(a: int, b: int): int requires true { a + b }\n"
        "method M(n: int) returns (r: int) requires n >= 0 ensures r == n { r := n; }"
    )
    assert header_violations(answer, spec) == ("header changed: Add",)


def test_violations_follow_spec_order() -> None:
    spec = "method B(n: int) returns (r: int) { r := n; }\nmethod A(n: int) returns (r: int) { r := n; }"
    answer = "method C(n: int) returns (r: int) { r := n; }"
    assert header_violations(answer, spec) == ("header missing: B", "header missing: A")


def test_extra_helper_lemma_before_method_is_allowed() -> None:
    answer = "lemma Helper(x: int) ensures x == x {} " + SAME
    assert header_violations(answer, SPEC) == ()


def test_extra_helper_function_after_method_is_allowed() -> None:
    answer = SAME + " function Twice(x: int): int { 2 * x }"
    assert header_violations(answer, SPEC) == ()


def test_trigger_attribute_inside_ensures_is_part_of_header() -> None:
    spec = "method M(xs: seq<int>) returns (r: int)\n  ensures forall x :: x in xs ==> r >= x\n{ r := 0; }"
    same = spec.replace("r := 0;", "r := 1;")
    assert header_violations(same, spec) == ()
    changed = spec.replace("forall x :: x in xs", "forall x {:trigger x in xs} :: x in xs")
    assert header_violations(changed, spec) == ("header changed: M",)


def test_trigger_attribute_kept_verbatim_gives_no_violation() -> None:
    spec = "method M(xs: seq<int>) ensures forall x {:trigger x in xs} :: x in xs ==> true { }"
    assert header_violations(spec, spec) == ()


def test_body_less_spec_declaration_needs_brace_in_answer() -> None:
    spec = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n"
    with_body = spec + " { r := n; }"
    assert header_violations(with_body, spec) == ()
    without_body = spec + ";"
    assert header_violations(without_body, spec) == ("header changed: M",)


def test_body_less_spec_declaration_with_semicolon() -> None:
    spec = "method M(n: int) returns (r: int);"
    answer = "method M(n: int) returns (r: int) { r := n; }"
    assert header_violations(answer, spec) == ()


def test_duplicate_declaration_is_changed_once() -> None:
    answer = SAME + " method M(n: int) returns (r: int) requires n >= 0 ensures r == n { r := n; }"
    assert header_violations(answer, SPEC) == ("header changed: M",)


def test_duplicate_declaration_with_changed_copy_is_changed_once() -> None:
    changed = SAME.replace("ensures r == n", "ensures r == n requires false", 1)
    assert header_violations(SAME + " " + changed, SPEC) == ("header changed: M",)


def test_unbalanced_braces_in_answer_is_changed() -> None:
    answer = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n { r := n; "
    assert header_violations(answer, SPEC) == ("header changed: M",)


def test_added_attribute_between_header_and_body_is_changed() -> None:
    answer = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n {:fuel 2} { r := n; }"
    assert header_violations(answer, SPEC) == ("header changed: M",)


def test_modifiers_are_part_of_header() -> None:
    spec = "static function F(x: int): int { x }"
    assert header_violations("static function F(x: int): int { x + 0 }", spec) == ()
    assert header_violations("function F(x: int): int { x }", spec) == ("header changed: F",)


def test_ghost_predicate_header() -> None:
    spec = "ghost predicate P(x: int) { x >= 0 }"
    assert header_violations("ghost predicate P(x: int) { x >= 0 }", spec) == ()
    assert header_violations("predicate P(x: int) { x >= 0 }", spec) == ("header changed: P",)


def test_lemma_declaration() -> None:
    spec = "lemma L(x: int) ensures x == x {}"
    assert header_violations("lemma L(x: int) ensures x == x { }", spec) == ()
    assert header_violations("method L(x: int) ensures x == x { }", spec) == ("header changed: L",)


def test_empty_spec_gives_no_violation() -> None:
    assert header_violations(SAME, "") == ()


def test_empty_answer_gives_missing() -> None:
    assert header_violations("", SPEC) == ("header missing: M",)


def test_keyword_inside_string_is_not_a_declaration() -> None:
    answer = 'method Note() returns (r: string) { r := "method M("; } ' + SAME
    assert header_violations(answer, SPEC) == ()


def test_keyword_inside_comment_is_not_a_declaration() -> None:
    answer = "// method M(n: int) returns (r: int)\n" + SAME
    assert header_violations(answer, SPEC) == ()


def test_lexer_is_reused_from_anti_cheat() -> None:
    """The header check reuses the anti-cheat scanner instead of copying it."""
    assert header_module._normalize is anti_cheat_module._normalize
    assert header_module._literal_spans is anti_cheat_module._literal_spans
    assert header_module._blanked is anti_cheat_module._blanked
    assert header_module._match_brace is anti_cheat_module._match_brace
    assert header_module._BRACE_ATTR_OPEN_RE is anti_cheat_module._BRACE_ATTR_OPEN_RE
