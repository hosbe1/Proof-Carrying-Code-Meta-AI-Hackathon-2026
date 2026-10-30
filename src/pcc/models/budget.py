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
    holds the usage of every completed call, in order.
    """

    def __init__(
        self,
        inner: ModelClient,
        *,
        max_cost_usd: float,
        max_calls: int,
        max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._max_cost_usd = max_cost_usd
        self._max_calls = max_calls
        self._max_seconds = max_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._usage: list[Usage] = []
        self._start: float | None = None

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
        with self._lock:
            if len(self._usage) >= self._max_calls:
                raise BudgetExceeded(f"budget exceeded: {len(self._usage)} calls reach max_calls={self._max_calls}")
            now = self._clock()
            if self._start is None:
                self._start = now
            elif now - self._start >= self._max_seconds:
                raise BudgetExceeded(f"budget exceeded: {now - self._start:.1f}s reach max_seconds={self._max_seconds}")
            spent = sum(usage.cost_usd for usage in self._usage)
            if spent >= self._max_cost_usd:
                raise BudgetExceeded(f"budget exceeded: ${spent:.4f} reaches max_cost_usd=${self._max_cost_usd:.4f}")
            completion = self._inner.complete(
                model, messages, step=step, schema=schema, temperature=temperature, max_tokens=max_tokens
            )
            self._usage.append(completion.usage)
            return completion

    @property
    def usage(self) -> tuple[Usage, ...]:
        """The usage of every completed call, in order."""
        with self._lock:
            return tuple(self._usage)


__all__ = ["BudgetExceeded", "BudgetedClient"]
