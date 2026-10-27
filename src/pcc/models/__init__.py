"""The only way to reach a language model.

Steps receive a ``ModelClient`` as an argument and never build one themselves, so tests run against
``FakeClient`` and CI never needs API keys. Real clients (Meta Model API, hosted Glimmer) live in this package too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pcc.core import Usage

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    usage: Usage


class ModelClient(Protocol):
    def complete(self, model: str, messages: tuple[Message, ...], *, step: str, temperature: float = 0.0) -> Completion:
        """Return one completion. ``step`` names the pipeline step, for cost reporting."""
        ...


class FakeClient:
    """Replays scripted answers in order and records every request. For tests only."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.requests: list[tuple[str, tuple[Message, ...], str]] = []

    def complete(self, model: str, messages: tuple[Message, ...], *, step: str, temperature: float = 0.0) -> Completion:
        self.requests.append((model, messages, step))
        if not self._answers:
            raise AssertionError(f"FakeClient: no scripted answer left for step {step!r}")
        return Completion(self._answers.pop(0), Usage(step=step, model=model))


__all__ = ["Completion", "FakeClient", "Message", "ModelClient", "Role"]
