"""Tests for the pull request comment (``pcc.report.render``).

The first three tests compare the whole comment against snapshots in ``tests/report/snapshots/``; the rest
cover ordering, the length limit, the time and cost formats, stale certificates, contradictions, the
second-opinion note and results without intent or translation. Every result is built with ``tests.helpers``.
"""

from __future__ import annotations

from pathlib import Path

from pcc.core import (
    AssumptionKind,
    CheckKind,
    Result,
    SourceKind,
    State,
    TranslationPath,
)
from pcc.report import render
from tests import helpers as h

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"


def read_snapshot(name: str) -> str:
    """The expected comment stored in ``tests/report/snapshots/<name>``."""
    return (SNAPSHOTS / name).read_text(encoding="utf-8")


def proven_result() -> Result:
    """One proven function with precondition, claims, assumptions and usage."""
    function = h.function_ref()
    return Result(
        function,
        State.PROVEN,
        "",
        h.intent(
            function,
            summary="Returns the total price of the cart.",
            claims=(
                h.claim(
                    "the total includes discounts",
                    (h.evidence(SourceKind.DOCSTRING, "README.md"),),
                ),
                h.claim(
                    "an empty cart totals 0",
                    (h.evidence(SourceKind.TEST, "tests/test_cart.py::test_empty"),),
                ),
            ),
        ),
        h.spec(
            function,
            predicate="def spec(self, result):\n    return result >= 0",
            precondition="len(self.items) > 0",
            checks=(h.check_record(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 93, 100),),
        ),
        h.translation(function),
        (
            h.assumption(AssumptionKind.LIBRARY_CONTRACT, "json.loads", "returns the parsed value"),
            h.assumption(AssumptionKind.UNPROVEN_CALLEE, "pricing.discount", "returns a non-negative discount"),
        ),
        None,
        h.proof(),
        None,
        (
            h.usage("prove", "muse-spark-1.3", 100, 50, 0.40, 180.0),
            h.usage("spec", "muse-spark-1.3", 10, 5, 0.02, 10.0),
        ),
    )


def counterexample_result() -> Result:
    """A counterexample with a ready-to-run test and a suggested fix."""
    function = h.function_ref("def checkout(cart): ...", path="src/shop.py", qualname="checkout", lineno=40)
    return Result(
        function,
        State.COUNTEREXAMPLE,
        "returned -1 instead of 0",
        h.intent(
            function,
            summary="Charges the cart.",
            claims=(
                h.claim(
                    "charges the sum of the prices",
                    (h.evidence(SourceKind.DOCSTRING, "src/shop.py"),),
                ),
                h.claim(
                    "an empty cart charges 0",
                    (h.evidence(SourceKind.TEST, "tests/test_shop.py::test_empty"),),
                ),
            ),
        ),
        h.spec(
            function,
            predicate="def spec(cart, result):\n    return result >= 0",
            precondition="len(cart.items) > 0",
            checks=(h.check_record(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 50, 100),),
        ),
        h.translation(function, function_path=TranslationPath.AI),
        (h.assumption(AssumptionKind.LIBRARY_CONTRACT, "stripe.charge", "charges the given amount"),),
        h.counterexample(
            arguments="[Item(3), Item(-1)]",
            expected="the sum of the prices",
            actual="-1",
            test="def test_checkout_breaks_spec():\n    assert checkout([Item(3), Item(-1)]) == 2",
            suggested_fix="--- a/src/shop.py\n+++ b/src/shop.py\n@@\n-    return -1\n+    return 0",
        ),
        None,
        None,
        (h.usage("counterexample", "muse-spark-1.3", 20, 10, 0.10, 30.0),),
    )


def test_render_one_proven_function_snapshot() -> None:
    assert render([proven_result()]) == read_snapshot("proven.md")


def test_render_counterexample_comes_first_snapshot() -> None:
    """The counterexample is rendered first even when it is passed last; one open ``pre`` warns below 20 %."""
    results = [open_discount_result(), open_total_result(), counterexample_result()]
    assert render(results) == read_snapshot("counterexample_and_open.md")


def open_total_result() -> Result:
    """An open function without a precondition."""
    total = h.function_ref("def total(self): ...", path="src/cart.py", qualname="Cart.total", lineno=12)
    return Result(
        total,
        State.OPEN,
        "budget exhausted",
        h.intent(total, summary="Returns the total price of the cart."),
        h.spec(total, precondition=None, checks=()),
        h.translation(total),
    )


def open_discount_result() -> Result:
    """An open function whose ``pre`` accepts 12 % of the generated inputs."""
    discount = h.function_ref("def discount(self): ...", path="src/cart.py", qualname="Cart.discount", lineno=30)
    return Result(
        discount,
        State.OPEN,
        "no proof found",
        h.intent(discount, summary="Returns the discount for the cart."),
        h.spec(
            discount,
            predicate="def spec(self, result):\n    return result >= 0",
            precondition="len(self.items) > 0",
            checks=(h.check_record(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 12, 100),),
        ),
        h.translation(discount),
    )


def test_render_many_proven_functions_are_shortened_snapshot() -> None:
    """400 proven functions do not fit; details are dropped and the note says how many."""
    actual = render(many_proven_results())
    assert len(actual) <= 65_000
    assert actual == read_snapshot("many_proven_shortened.md")


def many_proven_results() -> list[Result]:
    """400 proven functions that never fit into one comment."""
    results = []
    for i in range(400):
        function = h.function_ref(f"def f{i:03d}(): ...", path=f"src/mod{i:03d}.py", qualname=f"f{i:03d}", lineno=1)
        results.append(
            Result(
                function,
                State.PROVEN,
                f"proof {i:03d}",
                h.intent(function),
                h.spec(function),
                h.translation(function),
                (),
                None,
                h.proof(),
            )
        )
    return results


def test_render_no_results() -> None:
    assert render([]) == "<!-- pcc-check -->\n## Proof-carrying code\n\nNo functions checked.\n\nCost $0.00 · 0 s"


def test_render_summary_counts_in_fixed_order() -> None:
    """Zero counts are left out; the order is proven, counterexample, open, not checkable."""
    total = h.function_ref()
    results = [
        Result(total, State.NOT_CHECKABLE, "no type hints", None, None, None),
        Result(
            h.function_ref("def g(): ...", path="src/other.py", qualname="g", lineno=3),
            State.OPEN,
            "no proof found",
            None,
            None,
            None,
        ),
    ]
    assert "1 open, 1 not checkable" in render(results)


def test_render_not_checkable_without_intent_or_translation() -> None:
    """An early ``not_checkable`` result carries no intent and no translation; its reason renders unchanged."""
    reason = "cannot check `foo|bar` yet"
    actual = render([Result(h.function_ref(), State.NOT_CHECKABLE, reason, None, None, None)])
    assert "### ➖ `src/cart.py::Cart.total` — not checkable" in actual  # noqa: RUF001 - mandated icon
    assert reason in actual
    assert "translated by" not in actual
    assert "Intent:" not in actual


def test_render_cost_and_time_formats() -> None:
    function = h.function_ref()
    cents = Result(
        function,
        State.OPEN,
        "no proof found",
        None,
        None,
        None,
        (),
        None,
        None,
        None,
        (h.usage("prove", "muse-spark-1.3", 10, 5, 0.42, 190.0),),
    )
    assert render([cents]).splitlines()[-1] == "Cost $0.42 · 3 min 10 s"
    tiny = Result(
        function,
        State.OPEN,
        "no proof found",
        None,
        None,
        None,
        (),
        None,
        None,
        None,
        (h.usage("prove", "muse-spark-1.3", 1, 1, 0.004, 42.0),),
    )
    assert render([tiny]).splitlines()[-1] == "Cost <$0.01 · 42 s"


def test_render_stale_certificates_before_cost_line() -> None:
    stale = h.certificate(h.function_ref("def old(): ...", path="src/old.py", qualname="old", lineno=7))
    actual = render([proven_result()], stale=[stale])
    assert "## Stale certificates" in actual
    assert "`src/old.py::old`" in actual
    assert actual.index("## Stale certificates") < actual.index("Cost $0.42")


def test_render_contradiction_and_second_opinion() -> None:
    """Contradictions render with their resolution; the second opinion is one line."""
    function = h.function_ref()
    actual = render(
        [
            Result(
                function,
                State.OPEN,
                "no proof found",
                h.intent(
                    function,
                    contradictions=(
                        h.contradiction(),
                        h.contradiction("the name suggests rounding", (), None),
                    ),
                ),
                h.spec(function),
                h.translation(function),
                (),
                None,
                None,
                h.second_opinion(note="agrees."),
            )
        ]
    )
    assert "the docstring and the tests disagree on empty carts (resolved by: test)" in actual
    assert "the name suggests rounding (unresolved)" in actual
    assert "Second opinion: agrees." in actual
