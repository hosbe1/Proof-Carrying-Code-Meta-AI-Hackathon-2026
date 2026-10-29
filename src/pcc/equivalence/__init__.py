"""Step 4, checks: is the Dafny translation the same function as the original? (docs/plan.md, step 4)"""

from __future__ import annotations

from pathlib import Path

from pcc.core import EquivalenceReport, FunctionRef, Translation
from pcc.equivalence.spec_check import check_spec
from pcc.models import ModelClient


def check(
    function: FunctionRef,
    translation: Translation,
    repo_root: Path,
    *,
    inputs: int,
    symbolic_seconds: float,
    client: ModelClient,
    model: str,
) -> EquivalenceReport:
    """Compile the translation back to Python, convert values between Python and Dafny runtime types, compare with
    the original on ``inputs`` random inputs plus edge cases that ``model`` proposes through ``client`` (equal results
    and equal errors), then run ``crosshair diffbehavior``. The original code runs from ``repo_root``, so its imports
    resolve as they do in the project. The first mismatch found is returned as a counterexample."""
    raise NotImplementedError


__all__ = ["check", "check_spec"]
