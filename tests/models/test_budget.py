"""Tests for the budgeted model client (issue #19)."""

from __future__ import annotations

import threading

import pytest

from pcc.core import ToolError, Usage
from pcc.models import Completion, FakeClient, Message, ModelClient
from pcc.models.budget import BudgetedClient, BudgetExceeded

MESSAGES = (Message("user", "hi"),)


class _CostClient:
    """A ModelClient that reports a fixed cost per call (FakeClient always costs 0)."""

    def __init__(self, cost_per_call: float) -> None:
        self.cost_per_call = cost_per_call
        self.calls = 0

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
        self.calls += 1
        return Completion("ok", Usage(step=step, model=model, cost_usd=self.cost_per_call))


def test_max_calls_stops_at_the_limit() -> None:
    client = BudgetedClient(FakeClient(["a", "b", "c"]), max_cost_usd=100.0, max_calls=2, max_seconds=3600.0)
    assert client.complete("m", MESSAGES, step="one").text == "a"
    assert client.complete("m", MESSAGES, step="two").text == "b"
    with pytest.raises(BudgetExceeded):
        client.complete("m", MESSAGES, step="three")
    assert [usage.step for usage in client.usage] == ["one", "two"]


def test_budget_exceeded_is_not_a_tool_error() -> None:
    assert issubclass(BudgetExceeded, Exception)
    assert not issubclass(BudgetExceeded, ToolError)


def test_crossing_call_completes_then_next_raises() -> None:
    inner = _CostClient(0.006)
    client = BudgetedClient(inner, max_cost_usd=0.01, max_calls=100, max_seconds=3600.0)
    client.complete("m", MESSAGES, step="one")
    client.complete("m", MESSAGES, step="two")
    assert inner.calls == 2
    with pytest.raises(BudgetExceeded, match="max_cost_usd"):
        client.complete("m", MESSAGES, step="three")
    assert inner.calls == 2
    assert len(client.usage) == 2


def test_nested_run_limit_stops_both_clients() -> None:
    inner = _CostClient(0.004)
    run = BudgetedClient(inner, max_cost_usd=0.01, max_calls=100, max_seconds=3600.0)
    function: ModelClient = BudgetedClient(run, max_cost_usd=100.0, max_calls=100, max_seconds=3600.0)
    for _ in range(3):
        function.complete("m", MESSAGES, step="prove")
    with pytest.raises(BudgetExceeded):
        function.complete("m", MESSAGES, step="prove")
    with pytest.raises(BudgetExceeded):
        run.complete("m", MESSAGES, step="prove")
    assert inner.calls == 3
    assert len(run.usage) == 3
    assert isinstance(function, BudgetedClient)
    assert len(function.usage) == 3


def test_time_limit_counts_from_first_call() -> None:
    now = [1000.0]

    def clock() -> float:
        return now[0]

    client = BudgetedClient(
        FakeClient(["a", "b", "c"]),
        max_cost_usd=100.0,
        max_calls=100,
        max_seconds=30.0,
        clock=clock,
    )
    now[0] += 1000.0
    client.complete("m", MESSAGES, step="one")
    now[0] += 10.0
    client.complete("m", MESSAGES, step="two")
    now[0] += 30.0
    with pytest.raises(BudgetExceeded, match="max_seconds"):
        client.complete("m", MESSAGES, step="three")


def test_usage_holds_all_calls_in_order() -> None:
    client = BudgetedClient(FakeClient(["a", "b"]), max_cost_usd=100.0, max_calls=100, max_seconds=3600.0)
    client.complete("m", MESSAGES, step="one")
    client.complete("n", MESSAGES, step="two")
    assert isinstance(client.usage, tuple)
    assert [(usage.model, usage.step) for usage in client.usage] == [("m", "one"), ("n", "two")]


def test_failed_call_records_no_usage() -> None:
    class _Boom:
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
            raise ToolError("down")

    client = BudgetedClient(_Boom(), max_cost_usd=100.0, max_calls=100, max_seconds=3600.0)
    with pytest.raises(ToolError, match="down"):
        client.complete("m", MESSAGES, step="one")
    assert client.usage == ()


def test_concurrent_calls_respect_max_calls() -> None:
    client = BudgetedClient(FakeClient(["ok"] * 20), max_cost_usd=100.0, max_calls=20, max_seconds=3600.0)
    lock = threading.Lock()
    successes = 0

    def worker() -> None:
        nonlocal successes
        for _ in range(10):
            try:
                client.complete("m", MESSAGES, step="prove")
            except BudgetExceeded:
                return
            with lock:
                successes += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert successes == 20
    assert len(client.usage) == 20
