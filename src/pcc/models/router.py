"""Which prover stages run for a function. Filled in by a later ticket."""

from __future__ import annotations

from dataclasses import dataclass

from pcc.config import Config


@dataclass(frozen=True, slots=True)
class Router:
    """Decides the prover stages for a function."""


def available_stages(config: Config, router: Router) -> Config:
    """The config with the prover stages ``router`` selects."""
    raise NotImplementedError


__all__ = ["Router", "available_stages"]
