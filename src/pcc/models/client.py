"""The model client protocol every step talks through. Real clients live in this package too."""

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
        """Return one completion. ``step`` names the pipeline step, for cost reporting."""
        ...


__all__ = ["Completion", "Message", "ModelClient", "Role"]
