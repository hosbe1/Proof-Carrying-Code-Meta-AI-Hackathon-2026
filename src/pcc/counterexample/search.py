"""Step 6, search: an input on which the real code breaks the spec. Filled in by a later ticket."""

from __future__ import annotations

from pathlib import Path

from pcc.config import CounterexampleConfig
from pcc.core import Counterexample, FunctionRef, Spec
from pcc.models import ModelClient


def search(
    function: FunctionRef,
    spec: Spec,
    repo_root: Path,
    *,
    failed_clause: str | None,
    client: ModelClient,
    model: str,
    config: CounterexampleConfig,
) -> Counterexample | None:
    """Hypothesis focused on ``failed_clause`` plus inputs proposed by the model, each executed for real from
    ``repo_root``. Returns ``None`` when nothing was found (the function then ends as ``open``)."""
    raise NotImplementedError


__all__ = ["search"]
