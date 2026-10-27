"""Tunable settings, grouped by step so that parallel tickets edit different lines.

Each group is a frozen dataclass with defaults, and ``Config`` holds one of each: read a value as
``Config().proof.prover_stages``. Defaults are the starting values from docs/plan.md; they are revisited after the first
eval run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProverStage:
    name: str  # shown in results and reports, e.g. "glimmer"
    model: str  # model id as the model client understands it
    attempts: int


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    intent: str = "muse-spark"  # step 2
    spec: str = "muse-spark"  # step 3
    translation: str = "muse-spark"  # step 4, AI fallback


@dataclass(frozen=True, slots=True)
class TranspileConfig:
    python_version: str = "3.12"  # step 4: the one Python version the transpiler targets


@dataclass(frozen=True, slots=True)
class ProofConfig:
    prover_stages: tuple[ProverStage, ...] = (
        ProverStage("glimmer", "muse-glimmer-pcc", 8),
        ProverStage("spark", "muse-spark", 3),
        ProverStage("wonder", "muse-wonder", 1),
    )


@dataclass(frozen=True, slots=True)
class EquivalenceConfig:
    differential_inputs: int = 10_000  # step 4: random inputs for the differential test
    symbolic_seconds: float = 30.0  # step 4: time budget for crosshair diffbehavior
    max_retranslations: int = 3  # step 4: retranslations on the AI path after a mismatch


@dataclass(frozen=True, slots=True)
class DafnyConfig:
    version: str = "4.11.0"  # the release CI uses; its bundled Z3 decides the results
    resource_limit: int = 1_000_000  # dafny --resource-limit: deterministic, unlike a time limit


@dataclass(frozen=True, slots=True)
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    transpile: TranspileConfig = field(default_factory=TranspileConfig)
    proof: ProofConfig = field(default_factory=ProofConfig)
    equivalence: EquivalenceConfig = field(default_factory=EquivalenceConfig)
    dafny: DafnyConfig = field(default_factory=DafnyConfig)
