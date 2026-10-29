"""A scripted model client for tests. Real clients live in this package too."""

from __future__ import annotations

from dataclasses import dataclass

from pcc.core import Usage
from pcc.models.client import Completion, Message


@dataclass(frozen=True, slots=True)
class Call:
    """One recorded model call: which model, messages and step, plus the sampling options."""

    model: str
    messages: tuple[Message, ...]
    step: str
    schema: dict[str, object] | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


class FakeClient:
    """Replays scripted answers in order and records every request. For tests only."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.requests: list[tuple[str, tuple[Message, ...], str]] = []
        self.calls: list[Call] = []

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
        self.requests.append((model, messages, step))
        self.calls.append(Call(model, messages, step, schema, temperature, max_tokens))
        if not self._answers:
            raise AssertionError(f"FakeClient: no scripted answer left for step {step!r}")
        return Completion(self._answers.pop(0), Usage(step=step, model=model))


__all__ = ["Call", "FakeClient"]
