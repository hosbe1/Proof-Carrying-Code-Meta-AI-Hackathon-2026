from __future__ import annotations

import pytest

from pcc.models import FakeClient, Message


def test_replays_answers_in_order_and_records_requests() -> None:
    client = FakeClient(["first", "second"])
    messages = (Message("user", "hi"),)
    assert client.complete("m", messages, step="intent").text == "first"
    assert client.complete("m", messages, step="spec").text == "second"
    assert [step for _, _, step in client.requests] == ["intent", "spec"]


def test_fails_loudly_when_out_of_answers() -> None:
    with pytest.raises(AssertionError, match="no scripted answer"):
        FakeClient([]).complete("m", (), step="prove")
