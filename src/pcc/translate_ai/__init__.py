"""Step 4, fallback: translation by a model for whatever the transpiler does not support yet."""

from __future__ import annotations

from pcc.core import Counterexample, FunctionRef, Spec, Translation
from pcc.models import ModelClient


def translate(
    function: FunctionRef,
    spec: Spec,
    *,
    client: ModelClient,
    model: str,
    dafny_spec: str | None = None,
    mismatch: Counterexample | None = None,
) -> Translation:
    """Translate the function, and the spec to Dafny unless ``dafny_spec`` is given. A given ``dafny_spec`` was already
    translated by the transpiler: only the function is translated and the spec is kept as it is. ``mismatch`` is the
    counterexample of the previous attempt, if any."""
    raise NotImplementedError


__all__ = ["translate"]
