"""Tests for ``pcc.verifier.anti_cheat``: the scan that catches proof cheats Dafny accepts.

Dafny 4.11 verifies some cheats with exit code 0 (``{:axiom}``, ``@Axiom``, ``expect``,
``decreases *``, ``{:selective_checking}``), so the exit code alone cannot catch them.
The scan reports each violation as ``"line <n>: <form>"``; an empty tuple means the code passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcc.verifier.anti_cheat import ALLOWED_ATTRIBUTES, anti_cheat

CHEATS_DIR = Path(__file__).parent / "cheats"
CLEAN_DIR = Path(__file__).parent / "clean"

CHEAT_FIXTURES = sorted(path.name for path in CHEATS_DIR.glob("*.dfy"))


def test_allowlist_has_exactly_the_documented_names() -> None:
    assert ALLOWED_ATTRIBUTES == {
        "trigger",
        "induction",
        "fuel",
        "Fuel",
        "opaque",
        "split_here",
        "isolate_assertions",
        "IsolateAssertions",
    }


@pytest.mark.parametrize("fixture", CHEAT_FIXTURES)
def test_every_cheat_fixture_gives_at_least_one_violation(fixture: str) -> None:
    code = (CHEATS_DIR / fixture).read_text(encoding="utf-8")
    assert anti_cheat(code, "") != ()


def test_clean_fixture_passes() -> None:
    code = (CLEAN_DIR / "ok.dfy").read_text(encoding="utf-8")
    assert anti_cheat(code, "") == ()


def test_example_assume() -> None:
    assert anti_cheat("lemma L() ensures false { assume false; }", "") == ("line 1: assume",)


def test_example_frozen_spec_reindented_with_comment_passes() -> None:
    frozen_spec = "lemma Add(x: int) returns (r: int)\nensures r >= x"
    dafny_code = (
        "lemma   Add(x: int) returns (r: int)\n"
        "  // the sum is bigger than each summand\n"
        "  ensures r >= x\n"
        "{\n"
        "}"
    )
    assert anti_cheat(dafny_code, frozen_spec) == ()


def test_example_axiom_in_assumptions_module_passes() -> None:
    code = "module A { lemma {:axiom} C() ensures true }"
    assert anti_cheat(code, "", assumptions_module="A") == ()


def test_example_axiom_outside_assumptions_module_fails_twice() -> None:
    code = "module A { lemma {:axiom} C() ensures true }"
    assert anti_cheat(code, "", assumptions_module="") == (
        "line 1: body-less lemma",
        "line 1: axiom",
    )


def test_example_selective_checking() -> None:
    assert anti_cheat("lemma {:selective_checking} L() ensures false { }", "") == (
        "line 1: attribute selective_checking",
    )


def test_example_old_at_label_is_not_an_attribute() -> None:
    code = (
        "class C { var v: int }\n"
        "method m(c: C) modifies c { label L: c.v := c.v + 1; assert old@L(c.v) + 1 == c.v; }"
    )
    assert anti_cheat(code, "") == ()


def test_keyword_inside_identifier_is_no_violation() -> None:
    code = "method M() {\n  var assumed := 1;\n  var expectation := assumed + 1;\n  var lonely := 0;\n}"
    assert anti_cheat(code, "") == ()


def test_axiom_with_extra_attribute_outside_module() -> None:
    assert anti_cheat("lemma {:axiom} {:foo} L() ensures true", "") == (
        "line 1: body-less lemma",
        "line 1: axiom",
        "line 1: attribute foo",
    )


def test_at_axiom_on_line_before_lemma_outside_module() -> None:
    assert anti_cheat("@Axiom\nlemma L() ensures false", "") == (
        "line 1: axiom",
        "line 2: body-less lemma",
    )


def test_frozen_spec_with_changed_operator() -> None:
    code = "method M(x: int) returns (r: int) ensures x >= 0 {\n  r := x;\n}"
    assert anti_cheat(code, "ensures x > 0") == ("line 0: frozen spec changed",)


def test_frozen_spec_violation_sorts_before_line_violations() -> None:
    assert anti_cheat("method M() { assume false; }", "ensures true") == (
        "line 0: frozen spec changed",
        "line 1: assume",
    )


def test_empty_frozen_spec_skips_the_spec_check() -> None:
    assert anti_cheat("method M() { }", "") == ()


def test_verify_only_at_form() -> None:
    assert anti_cheat("@VerifyOnly\nlemma L() ensures false { }", "") == ("line 1: only",)


def test_only_brace_form_has_no_extra_attribute_violation() -> None:
    assert anti_cheat("lemma {:only} L(x: int) ensures x == x {\n}", "") == ("line 1: only",)


def test_verify_false_has_no_extra_attribute_violation() -> None:
    assert anti_cheat("lemma {:verify false} L(x: int) ensures false {\n}", "") == (
        "line 1: verify false",
    )


def test_extern_body_less_method() -> None:
    assert anti_cheat("method {:extern} M() ensures false", "") == (
        "line 1: body-less method",
        "line 1: extern",
    )


def test_expect_statement() -> None:
    assert anti_cheat("method M(x: int) {\n  expect x > 0;\n}", "") == ("line 2: expect",)


def test_decreases_star() -> None:
    assert anti_cheat("method M(n: int) decreases * { M(n); }", "") == ("line 1: decreases *",)


def test_decreases_multiplication_is_no_violation() -> None:
    assert anti_cheat("method M(n: int) decreases n * 2 { }", "") == ()


def test_termination_false() -> None:
    assert anti_cheat("function {:termination false} F(n: int): int { n }", "") == (
        "line 1: termination false",
    )


def test_resource_limit_fixture_reports_every_form() -> None:
    code = (CHEATS_DIR / "resource_limit.dfy").read_text(encoding="utf-8")
    assert anti_cheat(code, "") == (
        "line 2: resource limit override",
        "line 4: resource limit override",
        "line 6: resource limit override",
        "line 8: resource limit override",
        "line 10: resource limit override",
        "line 13: resource limit override",
        "line 16: resource limit override",
    )


def test_allowed_attributes_pass() -> None:
    code = (
        "function {:fuel 3} {:trigger P, Q} F(n: int): int { n }\n"
        "@IsolateAssertions\n"
        "lemma {:induction x} {:opaque} {:split_here} L(x: int) ensures x == x { }\n"
    )
    assert anti_cheat(code, "") == ()


def test_assume_and_expect_in_string_and_comment_are_no_violation() -> None:
    code = (
        '// assume false; expect false;\n'
        'method M() {\n'
        '  var s := "assume expect {:axiom} decreases *";\n'
        "}"
    )
    assert anti_cheat(code, "") == ()


def test_nested_block_comment_hides_cheat() -> None:
    assert anti_cheat("/* outer /* assume */ still comment */\nmethod M() { }", "") == ()


def test_char_literal_and_quoted_identifier() -> None:
    assert anti_cheat("method M() {\n  var s' := 'a';\n}", "") == ()


def test_unbalanced_open_brace() -> None:
    assert anti_cheat("method M() {", "") == ("line 1: unbalanced braces",)


def test_unbalanced_close_brace_reports_its_line() -> None:
    assert anti_cheat("method M() { }\n}", "") == ("line 2: unbalanced braces",)


def test_unmarked_body_less_lemma_inside_assumptions_module_still_fails() -> None:
    code = "module A { lemma L() ensures true }"
    assert anti_cheat(code, "", assumptions_module="A") == ("line 1: body-less lemma",)


def test_axiom_on_bodied_member_inside_assumptions_module_passes() -> None:
    code = "module A { lemma {:axiom} L() ensures true { } }"
    assert anti_cheat(code, "", assumptions_module="A") == ()


def test_at_axiom_before_lemma_inside_assumptions_module_passes() -> None:
    code = "module A {\n@Axiom\nlemma L() ensures true\n}"
    assert anti_cheat(code, "", assumptions_module="A") == ()


def test_extra_attribute_inside_assumptions_module_still_fails() -> None:
    code = "module A { lemma {:axiom} {:foo} L() ensures true }"
    assert anti_cheat(code, "", assumptions_module="A") == ("line 1: attribute foo",)


def test_nested_assumptions_module_passes() -> None:
    code = "module Outer { module A { lemma {:axiom} L() ensures true } }"
    assert anti_cheat(code, "", assumptions_module="A") == ()


def test_body_less_function_and_method() -> None:
    assert anti_cheat("function F(n: int): int", "") == ("line 1: body-less function",)
    assert anti_cheat("method M(n: int)", "") == ("line 1: body-less method",)


def test_any_input_gives_a_result() -> None:
    assert anti_cheat("", "") == ()
    assert anti_cheat("{{{", "???") != ()
