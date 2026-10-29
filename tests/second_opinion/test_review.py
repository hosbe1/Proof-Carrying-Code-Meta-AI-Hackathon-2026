"""Step 7: a second spec with the body in view, compared with the blind one."""

from __future__ import annotations

import json

import pytest

from pcc.core import Context, Evidence, SourceKind, Spec
from pcc.core.errors import ToolError
from pcc.models import Completion, FakeClient, Message, ModelClient
from pcc.second_opinion import review
from tests.helpers import function_ref

FUNCTION_SOURCE = "def maximum(xs: list[int]) -> int:\n    return max(xs)\n"
BLIND_SPEC = "def spec(xs, result):\n    return result in xs and all(x <= result for x in xs)\n"
ALTERNATIVE = "def spec(xs, result):\n    return result == max(xs)\n"


def make_context(source: str = FUNCTION_SOURCE) -> Context:
    function = function_ref(source=source, path="src/shop.py", qualname="maximum", lineno=3)
    evidence = (Evidence(SourceKind.DOCSTRING, "maximum:3", "Return the largest element."),)
    return Context(function=function, evidence=evidence)


def make_spec() -> Spec:
    return Spec(function=make_context().function, predicate=BLIND_SPEC)


def test_agree_answer_returns_no_alternative() -> None:
    client = FakeClient([json.dumps({"agrees": True, "alternative_predicate": None, "note": "matches blind spec"})])
    opinion = review(make_context(), make_spec(), client=client, model="muse-spark-1.3")

    assert opinion.agrees is True
    assert opinion.alternative_predicate is None
    assert opinion.note == "matches blind spec"


def test_disagree_answer_returns_the_alternative() -> None:
    client = FakeClient([json.dumps({"agrees": False, "alternative_predicate": ALTERNATIVE, "note": "body uses max"})])
    opinion = review(make_context(), make_spec(), client=client, model="muse-spark-1.3")

    assert opinion.agrees is False
    assert opinion.alternative_predicate == ALTERNATIVE
    assert opinion.note == "body uses max"


def test_alternative_without_def_spec_is_dropped_and_noted() -> None:
    bad = "def not_a_spec(xs, result):\n    return True\n"
    client = FakeClient([json.dumps({"agrees": False, "alternative_predicate": bad, "note": "try this instead"})])
    opinion = review(make_context(), make_spec(), client=client, model="muse-spark-1.3")

    assert opinion.agrees is False
    assert opinion.alternative_predicate is None
    assert "def spec" in opinion.note
    assert "dropped" in opinion.note.lower()


@pytest.mark.parametrize("note", ["", "   "])
def test_empty_note_becomes_no_note(note: str) -> None:
    client = FakeClient([json.dumps({"agrees": True, "alternative_predicate": None, "note": note})])
    opinion = review(make_context(), make_spec(), client=client, model="muse-spark-1.3")

    assert opinion.note == "no note"


def test_single_call_uses_second_opinion_step_and_sees_body_and_blind_spec() -> None:
    client = FakeClient([json.dumps({"agrees": True, "alternative_predicate": None, "note": "ok"})])
    review(make_context(), make_spec(), client=client, model="muse-spark-1.3")

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.step == "second_opinion"
    assert call.model == "muse-spark-1.3"
    assert call.schema is not None
    prompt = "\n".join(message.content for message in call.messages if isinstance(message, Message))
    assert FUNCTION_SOURCE in prompt
    assert BLIND_SPEC in prompt
    assert "Return the largest element." in prompt


def test_tool_error_passes_through() -> None:
    class BrokenClient(ModelClient):
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

    with pytest.raises(ToolError, match="model API down"):
        review(make_context(), make_spec(), client=BrokenClient(), model="muse-spark-1.3")
