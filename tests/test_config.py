"""Defaults of every config group (docs/plan.md starting values) and ProverStage validation."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pcc.config import (
    BudgetConfig,
    Config,
    CounterexampleConfig,
    DafnyConfig,
    EquivalenceConfig,
    ModelsConfig,
    PipelineConfig,
    ProofConfig,
    ProverStage,
    SpecConfig,
    TranspileConfig,
)


def test_models_use_the_spark_default_id() -> None:
    models = ModelsConfig()
    assert models.intent == "muse-spark-1.3"
    assert models.spec == "muse-spark-1.3"
    assert models.mutants == "muse-spark-1.3"
    assert models.translation == "muse-spark-1.3"
    assert models.edge_cases == "muse-spark-1.3"
    assert models.counterexample == "muse-spark-1.3"
    assert models.second_opinion == "muse-spark-1.3"
    assert models.context_search == "muse-spark-1.3"


def test_glimmer_endpoint_price_is_zero_per_token() -> None:
    assert Config().models.endpoint_prices["glimmer"] == (0.0, 0.0)


def test_proof_stages_default_to_glimmer_spark_wonder() -> None:
    stages = Config().proof.prover_stages
    assert [(stage.name, stage.model, stage.attempts) for stage in stages] == [
        ("glimmer", "glimmer", 8),
        ("spark", "muse-spark-1.3", 3),
        ("wonder", "muse-wonder-1.0", 1),
    ]
    assert [stage.endpoint for stage in stages] == ["glimmer", "meta", "meta"]
    assert [stage.max_source_lines for stage in stages] == [None, None, 40]


def test_prover_stage_defaults() -> None:
    stage = ProverStage("spark", "muse-spark-1.3", 3)
    assert stage.endpoint == "meta"
    assert stage.max_source_lines is None


def test_prover_stage_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="attempts"):
        ProverStage("spark", "muse-spark-1.3", 0)


def test_prover_stage_rejects_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        ProverStage("spark", "muse-spark-1.3", 3, endpoint="parallel")


def test_budget_defaults() -> None:
    budget = Config().budget
    assert budget.max_cost_usd_per_function == 1.0
    assert budget.max_seconds_per_function == 900.0
    assert budget.max_calls_per_function == 60
    assert budget.max_cost_usd_per_run == 20.0


def test_spec_defaults() -> None:
    spec = Config().spec
    assert spec.hypothesis_examples == 500
    assert spec.mutant_inputs == 200
    assert spec.max_rule_mutants == 30
    assert spec.model_mutants == 5
    assert spec.min_kill_ratio == 0.9
    assert spec.max_rewrites == 2


def test_counterexample_defaults() -> None:
    counterexample = Config().counterexample
    assert counterexample.hypothesis_examples == 5000
    assert counterexample.model_inputs == 10


def test_transpile_defaults() -> None:
    transpile = Config().transpile
    assert transpile.python_version == "3.12"
    assert transpile.ai_resolve_retries == 2


def test_dafny_defaults() -> None:
    dafny = Config().dafny
    assert dafny.version == "4.11.0"
    assert dafny.resource_limit == 1_000_000
    assert dafny.timeout_seconds == 600.0


def test_pipeline_defaults() -> None:
    pipeline = Config().pipeline
    assert pipeline.workers == 4
    assert pipeline.search_agent is False


def test_config_holds_one_group_of_each_kind() -> None:
    config = Config()
    assert isinstance(config.models, ModelsConfig)
    assert isinstance(config.proof, ProofConfig)
    assert isinstance(config.budget, BudgetConfig)
    assert isinstance(config.spec, SpecConfig)
    assert isinstance(config.counterexample, CounterexampleConfig)
    assert isinstance(config.transpile, TranspileConfig)
    assert isinstance(config.equivalence, EquivalenceConfig)
    assert isinstance(config.dafny, DafnyConfig)
    assert isinstance(config.pipeline, PipelineConfig)


@pytest.mark.parametrize(
    ("group", "plan_step"),
    [
        (ModelsConfig, "Models"),
        (ProverStage, "step 5"),
        (ProofConfig, "step 5"),
        (BudgetConfig, "Budget"),
        (SpecConfig, "step 3"),
        (CounterexampleConfig, "step 6"),
        (TranspileConfig, "step 4"),
        (EquivalenceConfig, "step 4"),
        (DafnyConfig, "step 5"),
        (PipelineConfig, "pipeline"),
    ],
)
def test_each_group_names_its_plan_step(group: type[object], plan_step: str) -> None:
    source = inspect.getsource(group)
    assert plan_step in source


def test_config_source_mentions_every_plan_section() -> None:
    source = Path(inspect.getfile(Config)).read_text()
    for plan_step in ("step 1", "step 2", "step 3", "step 4", "step 5", "step 6", "step 7", "Budget", "Models"):
        assert plan_step in source
