"""Step 3, checks: the spec against the original code and against mutants. Filled in by a later ticket."""

from __future__ import annotations

from pathlib import Path

from pcc.config import SpecConfig
from pcc.core import FunctionRef, Spec
from pcc.models import ModelClient


def check(
    spec: Spec, function: FunctionRef, repo_root: Path, *, client: ModelClient, model: str, config: SpecConfig
) -> Spec:
    """Test the spec against the original code (Hypothesis) and against mutants (fixed rules plus realistic bugs from
    the model, equivalent mutants filtered out). The original code runs from ``repo_root``, so its imports resolve as
    they do in the project. Return the spec with the check records added; after this the spec is frozen."""
    raise NotImplementedError


def spec_cases(
    spec: Spec, function: FunctionRef, repo_root: Path, *, config: SpecConfig
) -> tuple[tuple[str, str], ...]:
    """The inputs the spec translation check replays: ``(arguments, expected)`` pairs from the spec checks."""
    raise NotImplementedError


__all__ = ["check", "spec_cases"]
