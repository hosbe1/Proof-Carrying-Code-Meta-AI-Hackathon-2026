"""Step 3, writing: the spec as an executable Python predicate. Filled in by a later ticket."""

from __future__ import annotations

from pcc.core import Intent, Spec
from pcc.models import ModelClient


def write(intent: Intent, *, client: ModelClient, model: str, hint: str = "") -> Spec:
    """Turn the intent into ``spec(<args>, result) -> bool`` and an optional ``pre(<args>) -> bool``.

    The function body is never shown to the model. The prompt holds only the function's name, signature, type hints
    and the intent, even though ``intent.function.source`` contains the body. The test requirement is in
    docs/architecture.md, "Data flow for one function". ``hint`` holds only inputs with the original's and a mutant's
    result, never mutant code.
    """
    raise NotImplementedError


__all__ = ["write"]
