"""Tests for step 3, writing the spec blind from the intent (issue #26)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pcc.core import Claim, Evidence, Intent, NotCheckable, SourceKind, ToolError
from pcc.models import Completion, FakeClient, Message
from pcc.spec.prompts import SCHEMA, SPEC_CONSTRUCTS, build_user_prompt
from pcc.spec.write import write
from tests.helpers import function_ref

SOURCE = "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]  # SENTINEL-9d2c1-body"


def make_intent(source: str = SOURCE) -> Intent:
    function = function_ref(source=source, path="src/strings.py", qualname="is_palindrome", lineno=3)
    return Intent(
        function,
        "Decide whether a string reads the same forwards and backwards.",
        (
            Claim(
                "The result is true exactly for palindromes.",
                (Evidence(SourceKind.DOCSTRING, "is_palindrome docstring", "Return True for palindromes."),),
            ),
            Claim(
                "The empty string counts as a palindrome.",
                (Evidence(SourceKind.TEST, "tests/test_strings.py::test_empty", "is_palindrome('') is True"),),
            ),
        ),
        (),
    )


def answer(predicate: Any, precondition: Any = None, sources: Any = ()) -> str:
    return json.dumps({"predicate": predicate, "precondition": precondition, "precondition_sources": list(sources)})


PREDICATE = "def spec(s, result):\n    return result == (s == s[::-1])"
PRECONDITION = "def pre(s):\n    return len(s) >= 0"


def test_scripted_spec_returns_spec_and_step_is_spec() -> None:
    intent = make_intent()
    client = FakeClient([answer(PREDICATE)])
    spec = write(intent, client=client, model="m")

    assert client.requests[0][2] == "spec"
    assert spec.function is intent.function
    assert spec.predicate == PREDICATE
    assert spec.precondition is None
    assert spec.precondition_sources == ()
    assert spec.checks == ()


def test_schema_is_sent_with_the_call() -> None:
    intent = make_intent()
    client = FakeClient([answer(PREDICATE)])
    write(intent, client=client, model="m")

    assert client.calls[0].schema == SCHEMA
    assert SCHEMA["required"] == ["predicate", "precondition", "precondition_sources"]
    assert SCHEMA["additionalProperties"] is False


def test_missing_result_parameter_raises_after_retry() -> None:
    intent = make_intent()
    bad = "def spec(s):\n    return s == s[::-1]"
    client = FakeClient([answer(bad), answer(bad)])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")
    assert len(client.calls) == 2
    assert all(call.step == "spec" for call in client.calls)
    assert all(call.schema == SCHEMA for call in client.calls)


def test_invalid_first_answer_recovers_on_retry() -> None:
    intent = make_intent()
    client = FakeClient([answer("def spec(s):\n    return True"), answer(PREDICATE)])
    spec = write(intent, client=client, model="m")

    assert len(client.calls) == 2
    assert spec.predicate == PREDICATE
    retry_message = client.calls[1].messages[-1]
    assert retry_message.role == "user"
    assert "spec parameters must be" in retry_message.content


def test_retry_keeps_the_original_blind_prompt() -> None:
    intent = make_intent()
    client = FakeClient([answer("def spec(s):\n    return True"), answer(PREDICATE)])
    write(intent, client=client, model="m")

    first, second = client.calls[0].messages, client.calls[1].messages
    assert second[:2] == first
    assert SOURCE not in second[2].content


def test_hint_appears_verbatim_in_both_attempts() -> None:
    intent = make_intent()
    hint = "the spec transpiler does not support: dict display"
    client = FakeClient([answer("def spec(s):\n    return True"), answer(PREDICATE)])
    write(intent, client=client, model="m", hint=hint)

    for call in client.calls:
        assert hint in "\n".join(message.content for message in call.messages)


def test_pre_with_sources_maps_to_claims() -> None:
    intent = make_intent()
    client = FakeClient([answer(PREDICATE, PRECONDITION, [2])])
    spec = write(intent, client=client, model="m")

    assert spec.precondition == PRECONDITION
    assert spec.precondition_sources == (intent.claims[1],)


def test_pre_sources_come_back_in_claim_order() -> None:
    intent = make_intent()
    client = FakeClient([answer(PREDICATE, PRECONDITION, [2, 1])])
    spec = write(intent, client=client, model="m")

    assert spec.precondition_sources == (intent.claims[0], intent.claims[1])


def test_predicate_with_docstring_is_allowed() -> None:
    intent = make_intent()
    predicate = 'def spec(s, result):\n    """True exactly for palindromes."""\n    return result == (s == s[::-1])'
    client = FakeClient([answer(predicate)])
    spec = write(intent, client=client, model="m")

    assert spec.predicate == predicate


def test_lambda_in_predicate_is_rejected() -> None:
    intent = make_intent()
    bad = answer("def spec(s, result):\n    return (lambda x: x)(result)")
    client = FakeClient([bad, bad])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")


def test_malformed_json_gets_one_retry() -> None:
    intent = make_intent()
    client = FakeClient(["{not json", answer(PREDICATE)])
    spec = write(intent, client=client, model="m")

    assert len(client.calls) == 2
    assert spec.predicate == PREDICATE


def test_extra_key_gets_one_retry() -> None:
    intent = make_intent()
    extra = json.dumps(
        {
            "predicate": PREDICATE,
            "precondition": None,
            "precondition_sources": [],
            "note": "extra",
        }
    )
    client = FakeClient([extra, answer(PREDICATE)])
    spec = write(intent, client=client, model="m")

    assert spec.predicate == PREDICATE


def test_nonstring_predicate_gets_one_retry() -> None:
    intent = make_intent()
    client = FakeClient([answer(42), answer(PREDICATE)])
    spec = write(intent, client=client, model="m")

    assert spec.predicate == PREDICATE


def test_precondition_without_sources_is_rejected() -> None:
    intent = make_intent()
    bad = answer(PREDICATE, PRECONDITION, [])
    client = FakeClient([bad, bad])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")


def test_sources_without_precondition_are_rejected() -> None:
    intent = make_intent()
    bad = answer(PREDICATE, None, [1])
    client = FakeClient([bad, bad])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")


@pytest.mark.parametrize("sources", [[0], [3], [1, 1], [True], ["1"], [1, 9]])
def test_bad_source_indices_are_rejected(sources: list[Any]) -> None:
    intent = make_intent()
    bad = answer(PREDICATE, PRECONDITION, sources)
    client = FakeClient([bad, bad])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")
    assert len(client.calls) == 2


def test_function_source_appears_in_no_message() -> None:
    intent = make_intent()
    client = FakeClient([answer("def spec(s):\n    return True"), answer(PREDICATE, PRECONDITION, [1])])
    write(intent, client=client, model="m")

    assert len(client.calls) == 2
    for call in client.calls:
        for message in call.messages:
            assert SOURCE not in message.content


def test_signature_with_hints_reaches_the_prompt() -> None:
    intent = make_intent()
    client = FakeClient([answer(PREDICATE)])
    write(intent, client=client, model="m")

    user_text = "\n".join(message.content for message in client.calls[0].messages if message.role == "user")
    assert "is_palindrome(s: str) -> bool" in user_text
    assert "C1" in user_text and "C2" in user_text


def test_unsupported_signature_with_default_raises_without_model_call() -> None:
    intent = make_intent("def total(xs: list[int] = []) -> int:\n    return sum(xs)  # SENTINEL-9d2c1-body")
    client = FakeClient([])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")
    assert client.calls == []


def test_variadic_signature_raises_without_model_call() -> None:
    intent = make_intent("def total(*xs: int) -> int:\n    return sum(xs)  # SENTINEL-9d2c1-body")
    client = FakeClient([])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")
    assert client.calls == []


def test_unparseable_source_raises_without_model_call() -> None:
    intent = make_intent("def broken(:\n    pass")
    client = FakeClient([])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")
    assert client.calls == []


def test_in_on_list_param_is_allowed() -> None:
    intent = make_intent("def has(xs: list[int], v: int) -> bool:\n    return v in xs")
    predicate = "def spec(xs, v, result):\n    return result == (v in xs)"
    client = FakeClient([answer(predicate)])
    spec = write(intent, client=client, model="m")

    assert spec.predicate == predicate


def test_in_on_list_literal_is_allowed() -> None:
    intent = make_intent("def has(xs, v):\n    return v in xs")
    predicate = "def spec(xs, v, result):\n    return result == (v in [1, 2, 3])"
    client = FakeClient([answer(predicate)])
    spec = write(intent, client=client, model="m")

    assert spec.predicate == predicate


@pytest.mark.parametrize(
    "source",
    [
        "def has(s: str, v: str) -> bool:\n    return v in s",
        "def has(xs, v):\n    return v in xs",
    ],
)
def test_in_on_str_or_unknown_is_rejected(source: str) -> None:
    intent = make_intent(source)
    param = "s" if "(s: str" in source else "xs"
    bad = answer(f"def spec({param}, v, result):\n    return result == (v in {param})")
    client = FakeClient([bad, bad])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")


@pytest.mark.parametrize(
    "predicate",
    [
        "import os\ndef spec(s, result):\n    return True",
        "def spec(s, result):\n    for c in s:\n        pass\n    return True",
        "def spec(s, result):\n    x = True\n    return x",
        "def spec(s, result):\n    import os\n    return True",
        "def spec(s, result):\n    def helper():\n        return True\n    return helper()",
        "def other(s, result):\n    return True",
        "def spec(s, result):\n    return (yield True)",
        "def spec(s, result):\n    return result is True",
        "def spec(s, result):\n    return 2.5 == result",
        "def spec(s, result):\n    return all(x for x in 5)",
    ],
)
def test_forbidden_predicates_are_rejected(predicate: str) -> None:
    intent = make_intent()
    bad = answer(predicate)
    client = FakeClient([bad, bad])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")
    assert len(client.calls) == 2


def test_pre_with_result_parameter_is_rejected() -> None:
    intent = make_intent()
    bad = answer(PREDICATE, "def pre(s, result):\n    return True", [1])
    client = FakeClient([bad, bad])

    with pytest.raises(NotCheckable, match="spec malformed"):
        write(intent, client=client, model="m")


class _FailingClient:
    """A client whose tool is broken; write must let the error escape."""

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
        write(make_intent(), client=_FailingClient(), model="m")


def test_all_any_range_len_and_conditional_are_allowed() -> None:
    intent = make_intent("def maximum(xs: list[int]) -> int:\n    return max(xs)")
    predicate = (
        "def spec(xs, result):\n"
        "    return result in xs and all(x <= result for x in xs) and any(len(xs) > 0 for _ in range(1))"
    )
    client = FakeClient([answer(predicate)])
    spec = write(intent, client=client, model="m")

    assert spec.predicate == predicate


def test_spec_constructs_names_the_subset() -> None:
    for fragment in (
        "comparisons",
        "`and`",
        "`or`",
        "`not`",
        "`len`",
        "`all`",
        "`any`",
        "`range(...)`",
        "`in` on lists only",
        "conditional expressions",
    ):
        assert fragment in SPEC_CONSTRUCTS


def test_prompt_text_is_pinned() -> None:
    assert build_user_prompt(make_intent(), "is_palindrome(s: str) -> bool") == (
        "Function is_palindrome (src/strings.py:3).\n"
        "Signature: is_palindrome(s: str) -> bool\n"
        "Intent: Decide whether a string reads the same forwards and backwards.\n"
        "Claims:\n"
        "C1: The result is true exactly for palindromes.\n"
        "C2: The empty string counts as a palindrome.\n"
        "Allowed constructs:\n"
        f"{SPEC_CONSTRUCTS}\n"
        "Keep `pre` as weak as the intent allows; every clause needs a claim that demands it.\n"
        'Answer with the spec as JSON: {"predicate": "<def spec ...>", '
        '"precondition": "<def pre ...>" or null, '
        '"precondition_sources": [<1-based claim numbers>]}, with [] when there is no `pre`.'
    )
    assert build_user_prompt(make_intent(), "is_palindrome(s: str) -> bool", "stay blind") == (
        "Function is_palindrome (src/strings.py:3).\n"
        "Signature: is_palindrome(s: str) -> bool\n"
        "Intent: Decide whether a string reads the same forwards and backwards.\n"
        "Claims:\n"
        "C1: The result is true exactly for palindromes.\n"
        "C2: The empty string counts as a palindrome.\n"
        "Allowed constructs:\n"
        f"{SPEC_CONSTRUCTS}\n"
        "Hint: stay blind\n"
        "Keep `pre` as weak as the intent allows; every clause needs a claim that demands it.\n"
        'Answer with the spec as JSON: {"predicate": "<def spec ...>", '
        '"precondition": "<def pre ...>" or null, '
        '"precondition_sources": [<1-based claim numbers>]}, with [] when there is no `pre`.'
    )
