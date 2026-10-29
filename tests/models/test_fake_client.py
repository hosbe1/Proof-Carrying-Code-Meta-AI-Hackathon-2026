from __future__ import annotations

import pytest

from pcc.models import Call, FakeClient, Message


def test_replays_answers_in_order_and_records_requests() -> None:
    client = FakeClient(["first", "second"])
    messages = (Message("user", "hi"),)
    assert client.complete("m", messages, step="intent").text == "first"
    assert client.complete("m", messages, step="spec").text == "second"
    assert [step for _, _, step in client.requests] == ["intent", "spec"]


def test_fails_loudly_when_out_of_answers() -> None:
    with pytest.raises(AssertionError, match="no scripted answer"):
        FakeClient([]).complete("m", (), step="prove")


def test_records_schema_and_max_tokens_in_calls() -> None:
    client = FakeClient(["ok"])
    messages = (Message("user", "hi"),)
    schema: dict[str, object] = {"type": "object"}
    completion = client.complete("m", messages, step="spec", schema=schema, max_tokens=10)
    assert completion.text == "ok"
    assert client.calls == [Call("m", messages, "spec", schema, 0.0, 10)]
    assert client.requests == [("m", messages, "spec")]


def test_call_defaults_match_complete_defaults() -> None:
    client = FakeClient(["ok"])
    messages = (Message("user", "hi"),)
    client.complete("m", messages, step="spec", temperature=0.5)
    assert client.calls == [Call("m", messages, "spec", None, 0.5, None)]
    assert client.requests == [("m", messages, "spec")]
