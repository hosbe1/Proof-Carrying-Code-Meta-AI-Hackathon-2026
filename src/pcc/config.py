"""Tunable settings, grouped by step so that parallel tickets edit different lines.

Each group is a frozen dataclass with defaults, and ``Config`` holds one of each: read a value as
``Config().proof.prover_stages``. Defaults are the starting values from docs/plan.md; they are revisited after the first
eval run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ProverStage:  # step 5: one escalation stage of the proof agent
    name: str  # shown in results and reports, e.g. "glimmer"
    model: str  # model id as the model client understands it
    attempts: int  # step 5: proof attempts in this stage
    endpoint: str = "meta"  # step 5: which endpoint serves it, one of "meta", "glimmer"
    max_source_lines: int | None = None  # step 5: only run this stage below this size

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}")
        if self.endpoint not in ("meta", "glimmer"):
            raise ValueError(f"unknown endpoint: {self.endpoint!r}")


@dataclass(frozen=True, slots=True)
class ModelsConfig:  # Models: the model id used in each step
    intent: str = "muse-spark-1.3"  # step 2
    spec: str = "muse-spark-1.3"  # step 3
    mutants: str = "muse-spark-1.3"  # step 3: model-written mutants
    translation: str = "muse-spark-1.3"  # step 4, AI fallback
    edge_cases: str = "muse-spark-1.3"  # step 4: model-proposed edge-case inputs
    counterexample: str = "muse-spark-1.3"  # step 6
    second_opinion: str = "muse-spark-1.3"  # step 7
    context_search: str = "muse-spark-1.3"  # step 1: the search agent
    endpoint_prices: Mapping[str, tuple[float, float]] = MappingProxyType(  # Models: per 1M tokens by endpoint
        {"glimmer": (0.0, 0.0)}
    )


@dataclass(frozen=True, slots=True)
class TranspileConfig:  # step 4: the rule-based translation
    python_version: str = "3.12"  # step 4: the one Python version the transpiler targets
    ai_resolve_retries: int = 2  # step 4: AI retries for an unsupported construct


@dataclass(frozen=True, slots=True)
class ProofConfig:  # step 5: escalating prover stages, Glimmer first, then Spark, then Wonder
    prover_stages: tuple[ProverStage, ...] = (  # step 5: attempts per stage
        ProverStage("glimmer", "glimmer", 8, endpoint="glimmer"),
        ProverStage("spark", "muse-spark-1.3", 3),
        ProverStage("wonder", "muse-wonder-1.0", 1, max_source_lines=40),
    )


@dataclass(frozen=True, slots=True)
class EquivalenceConfig:  # step 4: checks that a translation matches the original
    differential_inputs: int = 10_000  # step 4: random inputs for the differential test
    symbolic_seconds: float = 30.0  # step 4: time budget for crosshair diffbehavior
    max_retranslations: int = 3  # step 4: retranslations on the AI path after a mismatch


@dataclass(frozen=True, slots=True)
class BudgetConfig:  # Budget: per-function and per-run spend, time and call limits
    max_cost_usd_per_function: float = 1.0  # Budget: spend cap for one function
    max_seconds_per_function: float = 900.0  # Budget: time cap for one function
    max_calls_per_function: int = 60  # Budget: model call cap for one function
    max_cost_usd_per_run: float = 20.0  # Budget: spend cap for a whole run


@dataclass(frozen=True, slots=True)
class SpecConfig:  # step 3: checks that the spec is tested and kills mutants
    hypothesis_examples: int = 500  # step 3: random inputs against the original
    mutant_inputs: int = 200  # step 3: inputs used to filter equivalent mutants
    max_rule_mutants: int = 30  # step 3: rule-based mutants the spec must kill
    model_mutants: int = 5  # step 3: model-written mutants the spec must kill
    min_kill_ratio: float = 0.9  # step 3: share of mutants the spec must reject
    max_rewrites: int = 2  # step 3: spec rewrites after a failed mutant check


@dataclass(frozen=True, slots=True)
class CounterexampleConfig:  # step 6: search for an input that breaks the spec
    hypothesis_examples: int = 5000  # step 6: random inputs focused on the unproven clause
    model_inputs: int = 10  # step 6: suspicious inputs proposed by the model


@dataclass(frozen=True, slots=True)
class DafnyConfig:  # step 5: the Dafny release CI verifies with
    version: str = "4.11.0"  # the release CI uses; its bundled Z3 decides the results
    resource_limit: int = 1_000_000  # dafny --resource-limit: deterministic, unlike a time limit
    timeout_seconds: float = 600.0  # step 5: wall-clock timeout for one dafny verify run


@dataclass(frozen=True, slots=True)
class PipelineConfig:  # pipeline: how steps 1-7 run, including the step 1 search agent
    workers: int = 4  # pipeline: parallel workers over functions
    search_agent: bool = True  # step 1: let the agent search further when the intent is unclear


@dataclass(frozen=True, slots=True)
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    transpile: TranspileConfig = field(default_factory=TranspileConfig)
    proof: ProofConfig = field(default_factory=ProofConfig)
    equivalence: EquivalenceConfig = field(default_factory=EquivalenceConfig)
    dafny: DafnyConfig = field(default_factory=DafnyConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    counterexample: CounterexampleConfig = field(default_factory=CounterexampleConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
