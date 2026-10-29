"""The ghost-only check on proof attempts. Filled in by a later ticket."""

from __future__ import annotations

from collections.abc import Sequence


def executable_part(dafny_method: str, *, lemma_names: Sequence[str]) -> str:
    """``dafny_method`` with all ghost code (lemmas and their calls) removed."""
    raise NotImplementedError


def same_executable(original: str, proven: str, *, lemma_names: Sequence[str]) -> bool:
    """True when ``proven`` differs from ``original`` only in ghost code, so the proof is about the same function."""
    raise NotImplementedError


__all__ = ["executable_part", "same_executable"]
