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
    dafny_spec: str,
    mismatch: Counterexample | None = None,
    resolve_retries: int = 2,
) -> Translation:
    """Translate the function; the spec is always translated by the transpiler and passed in as ``dafny_spec``.
    ``mismatch`` is the counterexample of the previous attempt, if any."""
    raise NotImplementedError


__all__ = ["translate"]
