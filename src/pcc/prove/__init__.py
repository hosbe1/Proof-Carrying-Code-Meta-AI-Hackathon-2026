"""Step 5: the proof agent with escalating prover stages (docs/plan.md, step 5)."""

from __future__ import annotations

from collections.abc import Sequence

from pcc.config import ProverStage
from pcc.core import Assumption, ProofOutcome, Translation
from pcc.models import ModelClient


def prove(
    translation: Translation,
    assumptions: Sequence[Assumption],
    *,
    client: ModelClient,
    stages: Sequence[ProverStage],
) -> ProofOutcome:
    """Add invariants, lemmas and hints until ``pcc.verifier.verify`` passes, feeding Dafny's errors back after each
    attempt. Every attempt goes through ``pcc.verifier.anti_cheat``; the spec stays frozen."""
    raise NotImplementedError


__all__ = ["prove"]
