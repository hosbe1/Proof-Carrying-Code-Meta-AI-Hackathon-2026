"""Per-function and per-run model budgets.

``BudgetedClient`` wraps a ``ModelClient`` and stops calls once a cost, call or time budget is used up. It is
thread-safe and can be nested, so a function-level client around a run-level client enforces both budgets. The
call that crosses a limit still completes; the next call raises ``BudgetExceeded``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from pcc.core import Usage
from pcc.models.client import Completion, Message, ModelClient


class BudgetExceeded(Exception):
    """A model budget (cost, calls or time) is used up. Not a ``ToolError``: the pipeline turns it into an
    ``open`` result with the reason "budget" instead of failing the run."""


class BudgetedClient:
    """A ``ModelClient`` wrapper that enforces a cost, call and time budget.

    The limits are checked before each call; the call that crosses a limit still completes, and the following
    call raises ``BudgetExceeded``. The time limit counts from the first call, not from construction. ``usage``
    holds the usage of every completed call, in order. With ``max_cost_per_call`` the caller supplies a
    conservative per-call cost bound. That amount is reserved before a call and released afterward, allowing
    concurrent calls while the total committed cost remains within ``max_cost_usd``. Without a bound, calls
    remain serialized so an unknown in-flight cost cannot overshoot the budget.
    """

    def __init__(
        self,
        inner: ModelClient,
        *,
        max_cost_usd: float,
        max_calls: int,
        max_seconds: float,
        max_cost_per_call: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_cost_per_call is not None and max_cost_per_call <= 0:
            raise ValueError("max_cost_per_call must be positive")
        self._inner = inner
        self._max_cost_usd = max_cost_usd
        self._max_calls = max_calls
        self._max_seconds = max_seconds
        self._max_cost_per_call = max_cost_per_call
        self._clock = clock
        self._lock = threading.Lock()
        self._usage: list[Usage] = []
        self._spent_cost = 0.0
        self._start: float | None = None
        self._in_flight = 0
        self._reserved_cost = 0.0

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
        """Run one completion unless a budget is already used up."""
        if self._max_cost_per_call is not None:
            reservation = self._max_cost_per_call
            with self._lock:
                if len(self._usage) + self._in_flight >= self._max_calls:
                    raise BudgetExceeded(f"budget exceeded: max_calls={self._max_calls}")
                now = self._clock()
                if self._start is None:
                    self._start = now
                elif now - self._start >= self._max_seconds:
                    raise BudgetExceeded(f"budget exceeded: max_seconds={self._max_seconds}")
                spent = self._spent_cost
                if spent + self._reserved_cost + reservation > self._max_cost_usd:
                    raise BudgetExceeded(
                        f"budget exceeded: ${spent + self._reserved_cost:.4f} committed and "
                        f"${reservation:.4f} needed reach max_cost_usd=${self._max_cost_usd:.4f}"
                    )
                self._in_flight += 1
                self._reserved_cost += reservation
            completion: Completion | None = None
            try:
                completion = self._inner.complete(
                    model, messages, step=step, schema=schema, temperature=temperature, max_tokens=max_tokens
                )
                return completion
            finally:
                with self._lock:
                    self._in_flight -= 1
                    self._reserved_cost -= reservation
                    if completion is not None:
                        self._usage.append(completion.usage)
                        self._spent_cost += completion.usage.cost_usd
        with self._lock:
            if len(self._usage) >= self._max_calls:
                raise BudgetExceeded(f"budget exceeded: {len(self._usage)} calls reach max_calls={self._max_calls}")
            now = self._clock()
            if self._start is None:
                self._start = now
            elif now - self._start >= self._max_seconds:
                raise BudgetExceeded(f"budget exceeded: {now - self._start:.1f}s reach max_seconds={self._max_seconds}")
            spent = self._spent_cost
            if spent >= self._max_cost_usd:
                raise BudgetExceeded(f"budget exceeded: ${spent:.4f} reaches max_cost_usd=${self._max_cost_usd:.4f}")
            completion = self._inner.complete(
                model, messages, step=step, schema=schema, temperature=temperature, max_tokens=max_tokens
            )
            self._usage.append(completion.usage)
            self._spent_cost += completion.usage.cost_usd
            return completion

    @property
    def usage(self) -> tuple[Usage, ...]:
        """The usage of every completed call, in order."""
        with self._lock:
            return tuple(self._usage)


__all__ = ["BudgetExceeded", "BudgetedClient"]
