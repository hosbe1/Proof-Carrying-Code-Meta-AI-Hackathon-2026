"""Step 3: the spec as an executable Python predicate, tested before anything is proven (docs/plan.md, step 3)."""

from __future__ import annotations

from pathlib import Path

from pcc.core import FunctionRef, Intent, Spec
from pcc.models import ModelClient


def write(intent: Intent, *, client: ModelClient, model: str) -> Spec:
    """Turn the intent into ``spec(<args>, result) -> bool`` and an optional ``pre(<args>) -> bool``.

    The function body is never shown to the model. The prompt holds only the function's name, signature, type hints
    and the intent, even though ``intent.function.source`` contains the body. The test requirement is in
    docs/architecture.md, "Data flow for one function"."""
    raise NotImplementedError


def check(spec: Spec, function: FunctionRef, repo_root: Path, *, client: ModelClient, model: str) -> Spec:
    """Test the spec against the original code (Hypothesis) and against mutants (fixed rules plus realistic bugs from
    the model, equivalent mutants filtered out). The original code runs from ``repo_root``, so its imports resolve as
    they do in the project. Return the spec with the check records added; after this the spec is frozen."""
    raise NotImplementedError


__all__ = ["check", "write"]
