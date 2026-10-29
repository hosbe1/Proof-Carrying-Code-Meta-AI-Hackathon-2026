"""Tests for step 2: deriving the intent from context evidence without seeing the body."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from pcc.core import Context, Evidence, NotCheckable, SourceKind, ToolError
from pcc.intent import derive
from pcc.intent.prompts import SCHEMA, SYSTEM_PROMPT, build_user_prompt
from pcc.models import Completion, FakeClient, Message
from tests.helpers import function_ref

SOURCE = "def maximum(xs: list[int]) -> int:\n    return xs[0]  # SENTINEL-7f3a9c-body"


def make_context() -> Context:
    function = function_ref(source=SOURCE, path="src/shop.py", qualname="maximum", lineno=7)
    evidence = (
        Evidence(SourceKind.TEST, "tests/test_shop.py::test_maximum", "maximum([3, 1, 2]) == 3"),
        Evidence(SourceKind.DOCSTRING, "maximum docstring", "Return the largest element."),
    )
    return Context(function, evidence)


def answer(
    claims: list[dict[str, Any]],
    contradictions: list[dict[str, Any]] | None = None,
    *,
    checkable: bool = True,
    reason: str = "",
    summary: str = "Return the largest element of a non-empty list.",
) -> str:
    return json.dumps(
        {
            "summary": summary,
            "claims": claims,
            "contradictions": contradictions if contradictions is not None else [],
            "checkable": checkable,
            "reason": reason,
        }
    )


def test_two_claims_resolve_to_evidence_objects_and_step_is_intent() -> None:
    context = make_context()
    client = FakeClient(
        [
            answer(
                [
                    {"text": "Returns the largest element.", "sources": [2]},
                    {"text": "Accepts a non-empty list.", "sources": [1]},
                ]
            )
        ]
    )
    intent = derive(context, client=client, model="m")

    assert client.requests[0][2] == "intent"
    assert intent.function is context.function
    assert intent.summary == "Return the largest element of a non-empty list."
    assert [claim.text for claim in intent.claims] == [
        "Returns the largest element.",
        "Accepts a non-empty list.",
    ]
    assert intent.claims[0].sources == (context.evidence[1],)
    assert intent.claims[1].sources == (context.evidence[0],)
    assert intent.contradictions == ()


def test_contradiction_between_test_and_docstring_resolves_to_test() -> None:
    context = make_context()
    client = FakeClient(
        [
            answer(
                [{"text": "Returns the largest element.", "sources": [1]}],
                [{"description": "Tests expect the largest, the docstring the smallest.", "sources": [1, 2]}],
            )
        ]
    )
    intent = derive(context, client=client, model="m")

    assert len(intent.contradictions) == 1
    contradiction = intent.contradictions[0]
    assert contradiction.sources == (context.evidence[0], context.evidence[1])
    assert contradiction.resolved_by is SourceKind.TEST


def test_contradiction_between_two_comments_has_no_winner() -> None:
    function = function_ref(source=SOURCE)
    context = Context(
        function,
        (
            Evidence(SourceKind.COMMENT, "src/shop.py:4", "largest element"),
            Evidence(SourceKind.COMMENT, "src/shop.py:9", "smallest element"),
        ),
    )
    client = FakeClient(
        [
            answer(
                [{"text": "Returns an extreme element.", "sources": [1]}],
                [{"description": "Two comments disagree on which extreme.", "sources": [1, 2]}],
            )
        ]
    )
    intent = derive(context, client=client, model="m")

    assert intent.contradictions[0].resolved_by is None


def test_contradiction_between_issue_and_pull_request_has_no_winner() -> None:
    function = function_ref(source=SOURCE)
    context = Context(
        function,
        (
            Evidence(SourceKind.ISSUE, "#12", "largest element"),
            Evidence(SourceKind.PULL_REQUEST, "#34", "smallest element"),
        ),
    )
    client = FakeClient(
        [
            answer(
                [{"text": "Returns an extreme element.", "sources": [1]}],
                [{"description": "Issue and pull request disagree.", "sources": [1, 2]}],
            )
        ]
    )
    intent = derive(context, client=client, model="m")

    assert intent.contradictions[0].resolved_by is None


def test_not_checkable_answer_raises_with_reason() -> None:
    context = make_context()
    client = FakeClient([answer([], [], checkable=False, reason="sources contradict")])

    with pytest.raises(NotCheckable, match="sources contradict"):
        derive(context, client=client, model="m")


def test_index_zero_gets_one_retry_naming_the_error() -> None:
    context = make_context()
    client = FakeClient(
        [
            answer([{"text": "Returns the largest element.", "sources": [0]}]),
            answer([{"text": "Returns the largest element.", "sources": [2]}]),
        ]
    )
    intent = derive(context, client=client, model="m")

    assert len(client.calls) == 2
    retry_message = client.calls[1].messages[-1]
    assert retry_message.role == "user"
    assert "E0" in retry_message.content
    assert intent.claims[0].sources == (context.evidence[1],)


def test_index_beyond_evidence_count_fails_after_retry() -> None:
    context = make_context()
    client = FakeClient(
        [
            answer([{"text": "Returns the largest element.", "sources": [9]}]),
            answer([{"text": "Returns the largest element.", "sources": [0]}]),
        ]
    )

    with pytest.raises(NotCheckable, match="intent without sources"):
        derive(context, client=client, model="m")
    assert len(client.calls) == 2


def test_function_source_appears_in_no_message() -> None:
    context = make_context()
    client = FakeClient([answer([{"text": "Returns the largest element.", "sources": [1, 2]}])])
    derive(context, client=client, model="m")

    assert client.calls
    for call in client.calls:
        for message in call.messages:
            assert SOURCE not in message.content


def test_schema_is_sent_with_the_call() -> None:
    context = make_context()
    client = FakeClient([answer([{"text": "Returns the largest element.", "sources": [1]}])])
    derive(context, client=client, model="m")

    assert client.calls[0].schema == SCHEMA
    required = cast(list[str], SCHEMA["required"])
    assert set(required) == {"summary", "claims", "contradictions", "checkable", "reason"}


class _FailingClient:
    """A client whose tool is broken; derive must let the error escape."""

    def complete(
        self,
        model: str,
        messages: tuple[Message, ...],
        *,
        step: str,
        schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        raise ToolError("model API down")


def test_tool_error_passes_through() -> None:
    with pytest.raises(ToolError, match="model API down"):
        derive(make_context(), client=_FailingClient(), model="m")


def test_prompt_text_is_pinned() -> None:
    assert SYSTEM_PROMPT == (
        "You describe what a Python function should do, in plain language.\n"
        "You never see the function body; work only from the numbered evidence below.\n"
        "Every claim must cite at least one evidence item by its number (E1, E2, ...).\n"
        "Flag every contradiction between evidence items.\n"
        'Answer with JSON only: {"summary": str, "claims": [{"text": str, "sources": [int]}], '
        '"contradictions": [{"description": str, "sources": [int]}], "checkable": bool, "reason": str}.\n'
        "Set checkable to false with a reason when the contradictions make any sensible intent impossible."
    )

    function = function_ref(source=SOURCE, path="src/cart.py", qualname="Cart.total", lineno=12)
    context = Context(
        function,
        (
            Evidence(SourceKind.TEST, "tests/test_cart.py::test_total", "total([2, 1]) == 2"),
            Evidence(SourceKind.DOCSTRING, "Cart.total docstring", "Return the total."),
        ),
    )
    assert build_user_prompt(context) == (
        "Function Cart.total (src/cart.py:12).\n"
        "Evidence:\n"
        "E1 [test] tests/test_cart.py::test_total: total([2, 1]) == 2\n"
        "E2 [docstring] Cart.total docstring: Return the total."
    )
