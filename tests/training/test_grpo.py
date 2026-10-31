"""Tests for the GRPO loop in ``training/grpo.py``.

Everything here runs offline with a fake ``Backend`` and a fake ``reward_fn``: no network, no GPU
and no Dafny. The dataset split and (in one test) the graded reward are stubbed with
``monkeypatch``; every other test passes an explicit fake ``reward_fn``.
"""

from __future__ import annotations

import ast
import dataclasses
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from training.grpo import (
    Backend,
    GrpoConfig,
    RunResult,
    Sample,
    TrainStats,
    estimate_spend,
    load_config,
    run,
)
from training.prompts import SYSTEM_MESSAGE, build_prompt

from pcc.config import Config
from pcc.core import ToolError
from pcc.evaluation import Task
from training import grpo

REPO_ROOT = Path(grpo.__file__).resolve().parent.parent
SHIPPED_CONFIG = REPO_ROOT / "training" / "configs" / "glimmer_fireworks.toml"


def good_reward(answer: str, task_spec: str) -> float:
    """Fake graded reward: 1.0 for a "good" attempt, 0.0 otherwise."""
    assert task_spec != ""
    return 1.0 if answer == "good" else 0.0


def make_task(name: str) -> Task:
    """A fixture training task whose spec carries ``name`` as its marker."""
    return Task(
        id=f"vericoding/{name}",
        source_task=f"vericoding/fixture/{name}",
        payload={
            "vc-preamble": "",
            "vc-helpers": "",
            "vc-spec": f"method {name}() returns (r: int) // {name}",
            "vc-code": "{ r := 0; }",
        },
    )


def make_config(**overrides: object) -> GrpoConfig:
    """A small deterministic config; tests override fields as needed."""
    base = GrpoConfig(
        model="test-model",
        prefill_usd_per_1m=1.96,
        cached_prefill_usd_per_1m=0.39,
        sample_usd_per_1m=4.88,
        train_usd_per_1m=5.86,
        max_spend_usd=1000.0,
        prepass_tasks=64,
        seed=0,
        tasks_per_step=16,
        samples_per_task=8,
        lora_rank=16,
        learning_rate=1e-5,
        base_adapter="",
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


class FakeBackend:
    """A scripted offline backend.

    ``goods`` maps a prompt marker to the number of "good" attempts each ``sample`` call
    returns for prompts containing that marker. The first ``prepass_sample_calls`` calls use the
    pre-pass token counts, later calls the step token counts. Call numbers in
    ``fail_sample_on`` / ``fail_train_on`` raise ``ToolError`` instead of answering.
    """

    def __init__(
        self,
        goods: Mapping[str, int],
        *,
        prepass_sample_calls: int = 0,
        pre_prompt_tokens: int = 10,
        pre_completion_tokens: int = 5,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        train_tokens: int = 1000,
        post_goods: int | None = None,
        fail_sample_on: frozenset[int] = frozenset(),
        fail_train_on: frozenset[int] = frozenset(),
    ) -> None:
        self._goods = dict(goods)
        self._prepass_sample_calls = prepass_sample_calls
        self._pre_sizes = (pre_prompt_tokens, pre_completion_tokens)
        self._sizes = (prompt_tokens, completion_tokens)
        self._train_tokens = train_tokens
        self._post_goods = post_goods
        self._fail_sample_on = fail_sample_on
        self._fail_train_on = fail_train_on
        self.sample_calls: list[tuple[str, int]] = []
        self.train_calls: list[list[tuple[str, str, float]]] = []
        self.save_calls = 0

    def _marker(self, prompt: str) -> str:
        for marker in self._goods:
            if marker in prompt:
                return marker
        raise AssertionError(f"prompt carries no known task marker: {prompt[:80]!r}")

    def sample(self, prompt: str, n: int) -> list[Sample]:
        self.sample_calls.append((prompt, n))
        if len(self.sample_calls) in self._fail_sample_on:
            raise ToolError("boom: sampler failed")
        if self._post_goods is not None and len(self.sample_calls) > self._prepass_sample_calls:
            good = self._post_goods
        else:
            good = self._goods[self._marker(prompt)]
        sizes = self._pre_sizes if len(self.sample_calls) <= self._prepass_sample_calls else self._sizes
        return [
            Sample(
                text="good" if i < good else "bad",
                prompt_tokens=sizes[0],
                completion_tokens=sizes[1],
            )
            for i in range(n)
        ]

    def train(self, batch: list[tuple[str, str, float]]) -> TrainStats:
        self.train_calls.append(list(batch))
        if len(self.train_calls) in self._fail_train_on:
            raise ToolError("boom: trainer failed")
        return TrainStats(tokens=self._train_tokens)

    def save(self) -> str:
        self.save_calls += 1
        return "adapter-1"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test forgets to stub the dataset split: no test may touch the network."""

    def _blocked(name: str, which: str = "train") -> tuple[Task, ...]:
        raise AssertionError("tests must stub grpo.split with fixture tasks")

    monkeypatch.setattr(grpo, "split", _blocked)


def _stub_split(monkeypatch: pytest.MonkeyPatch, tasks: list[Task]) -> list[tuple[str, str]]:
    """Stub ``split`` with fixture tasks and record how ``run`` calls it."""
    calls: list[tuple[str, str]] = []

    def _fake(name: str, which: str) -> tuple[Task, ...]:
        calls.append((name, which))
        return tuple(tasks)

    monkeypatch.setattr(grpo, "split", _fake)
    return calls


def test_estimate_spend_matches_example() -> None:
    config = make_config()
    assert estimate_spend(1_000_000, 1_000_000, 1_000_000, config) == pytest.approx(12.70)


def test_cached_prefill_rate_stays_unused() -> None:
    config = make_config()
    changed = dataclasses.replace(config, cached_prefill_usd_per_1m=99.0)
    assert estimate_spend(10_000, 20_000, 30_000, changed) == estimate_spend(10_000, 20_000, 30_000, config)


def test_shipped_config_loads() -> None:
    config = load_config(SHIPPED_CONFIG)
    assert config == GrpoConfig(
        model="accounts/fireworks/models/muse-glimmer-30b",
        prefill_usd_per_1m=1.96,
        cached_prefill_usd_per_1m=0.39,
        sample_usd_per_1m=4.88,
        train_usd_per_1m=5.86,
        max_spend_usd=50,
        prepass_tasks=64,
        seed=config.seed,
        tasks_per_step=16,
        samples_per_task=8,
        lora_rank=16,
        learning_rate=1e-5,
        base_adapter=config.base_adapter,
    )


def test_shipped_config_raw_values() -> None:
    raw = tomllib.loads(SHIPPED_CONFIG.read_bytes().decode("utf-8"))
    assert raw["model"] == "accounts/fireworks/models/muse-glimmer-30b"
    assert raw["max_spend_usd"] == 50
    assert (raw["prefill_usd_per_1m"], raw["sample_usd_per_1m"], raw["train_usd_per_1m"]) == (1.96, 4.88, 5.86)
    assert raw["cached_prefill_usd_per_1m"] == 0.39
    assert (raw["prepass_tasks"], raw["tasks_per_step"], raw["samples_per_task"]) == (64, 16, 8)
    assert (raw["lora_rank"], raw["learning_rate"]) == (16, 1e-5)
    assert isinstance(raw["seed"], int)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.toml")


def test_load_config_rejects_bad_toml(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("model = [\n", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(path)


def test_prompts_match_shared_build_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-PROMPT")]
    _stub_split(monkeypatch, tasks)
    backend = FakeBackend({"TASK-PROMPT": 1}, prepass_sample_calls=1)
    config = make_config(tasks_per_step=1, samples_per_task=2, prepass_tasks=1, max_spend_usd=0.01)
    run(config, backend, reward_fn=good_reward)
    expected = f"{SYSTEM_MESSAGE}\n\n{build_prompt(tasks[0])}"
    assert backend.sample_calls != []
    assert all(prompt == expected for prompt, _ in backend.sample_calls)


def test_group_filter_and_advantages_sum_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-ONLY")]
    split_calls = _stub_split(monkeypatch, tasks)
    backend = FakeBackend({"TASK-ONLY": 1}, prepass_sample_calls=1)
    # One pre-pass sample (10/5 tokens) costs 0.000044, so the first-step estimate is 0.005632;
    # one step (2 samples at 100/50 tokens plus 1000 train tokens) costs 0.00674, so exactly
    # one step fits into the cap.
    config = make_config(tasks_per_step=1, samples_per_task=2, prepass_tasks=1, max_spend_usd=0.01)
    result = run(config, backend, reward_fn=good_reward)
    assert split_calls == [("vericoding", "train")]
    # One group with rewards 0 and 1 is kept; its advantages sum to 0.
    assert len(backend.train_calls) == 1
    batch = backend.train_calls[0]
    assert len(batch) == 2
    assert [advantage for _, _, advantage in batch] == pytest.approx([1.0, -1.0], abs=1e-3)
    assert sum(advantage for _, _, advantage in batch) == 0.0
    assert all(isinstance(text, str) and isinstance(prompt, str) for prompt, text, _ in batch)
    assert result.steps_done == 1
    assert result.stop_reason == "spend cap"
    assert result.adapter_id == "adapter-1"
    assert backend.save_calls == 1


def test_all_equal_group_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-FLAT")]
    _stub_split(monkeypatch, tasks)
    backend = FakeBackend({"TASK-FLAT": 0})
    config = make_config(tasks_per_step=1, samples_per_task=4, prepass_tasks=1)
    result = run(config, backend, reward_fn=good_reward)
    # Pre-pass solves 0 of 4: nothing is kept, so no step samples and no train runs.
    assert backend.train_calls == []
    assert len(backend.sample_calls) == 1
    assert result.steps_done == 0
    assert result.adapter_id == "adapter-1"
    assert result.stop_reason == "all groups dropped"
    assert result.spend_usd == pytest.approx(estimate_spend(400, 200, 0, config))
    assert backend.save_calls == 1


def test_prepass_keeps_partial_solves_only(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-A"), make_task("TASK-B"), make_task("TASK-C")]
    _stub_split(monkeypatch, tasks)
    # TASK-A is solved 3 of 8 times (kept); TASK-B 0 of 8 and TASK-C 8 of 8 (dropped).
    # After the pre-pass every attempt fails, so the first step drops all groups and stops.
    backend = FakeBackend({"TASK-A": 3, "TASK-B": 0, "TASK-C": 8}, prepass_sample_calls=3, post_goods=0)
    config = make_config(samples_per_task=8)
    result = run(config, backend, reward_fn=good_reward)
    prepass_prompts = [prompt for prompt, _ in backend.sample_calls[:3]]
    markers = ("TASK-A", "TASK-B", "TASK-C")
    seen_markers = {marker for prompt in prepass_prompts for marker in markers if marker in prompt}
    assert seen_markers == set(markers)
    assert all(n == 8 for _, n in backend.sample_calls[:3])
    step_prompts = [prompt for prompt, _ in backend.sample_calls[3:]]
    assert step_prompts != []
    assert all("TASK-A" in prompt for prompt in step_prompts)
    assert all("TASK-B" not in prompt and "TASK-C" not in prompt for prompt in step_prompts)
    assert backend.train_calls == []
    assert result.stop_reason == "all groups dropped"


def test_two_steps_then_spend_cap_saves_once(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task(f"TASK-{i}") for i in range(4)]
    _stub_split(monkeypatch, tasks)
    goods = {f"TASK-{i}": 1 for i in range(4)}
    backend = FakeBackend(goods, prepass_sample_calls=4, pre_prompt_tokens=10, pre_completion_tokens=5)
    config = make_config(
        tasks_per_step=4,
        samples_per_task=2,
        max_spend_usd=0.025,
    )
    assert isinstance(backend, Backend)
    result = run(config, backend, reward_fn=good_reward)
    # Pre-pass (4 tasks x 2 samples at 10/5 tokens) costs 0.000352; each step
    # (4 tasks x 2 samples at 100/50 tokens plus 1000 train tokens) costs 0.00938.
    # The first-step estimate is 0.005632, so exactly two steps fit into $0.025.
    assert result.steps_done == 2
    assert result.stop_reason == "spend cap"
    assert result.adapter_id == "adapter-1"
    assert backend.save_calls == 1
    assert len(backend.train_calls) == 2
    assert len(backend.sample_calls) == 4 + 2 * 4
    assert result.spend_usd == pytest.approx(0.019112)
    assert result.spend_usd <= config.max_spend_usd
    for batch in backend.train_calls:
        assert len(batch) == 8
        first, second = batch[0][2], batch[1][2]
        assert first + second == 0.0


def test_prepass_alone_exceeding_cap_saves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-X")]
    _stub_split(monkeypatch, tasks)
    backend = FakeBackend({"TASK-X": 1}, prepass_sample_calls=1)
    config = make_config(max_spend_usd=1e-9, prepass_tasks=1, samples_per_task=2)
    result = run(config, backend, reward_fn=good_reward)
    assert result.steps_done == 0
    assert result.stop_reason == "spend cap"
    assert result.adapter_id is None
    assert backend.save_calls == 0
    assert backend.train_calls == []
    assert result.spend_usd > config.max_spend_usd


def test_step_exceeding_cap_is_not_started(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-Y")]
    _stub_split(monkeypatch, tasks)
    backend = FakeBackend({"TASK-Y": 1}, prepass_sample_calls=1)
    # The pre-pass fits, but the first-step estimate (128 x per-sample cost) does not.
    config = make_config(max_spend_usd=0.001, prepass_tasks=1, samples_per_task=2)
    result = run(config, backend, reward_fn=good_reward)
    assert result.steps_done == 0
    assert result.stop_reason == "spend cap"
    assert backend.train_calls == []
    assert len(backend.sample_calls) == 1


def test_tool_error_before_first_step_saves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-Z")]
    _stub_split(monkeypatch, tasks)
    backend = FakeBackend({"TASK-Z": 1}, fail_sample_on=frozenset({1}))
    config = make_config(prepass_tasks=1, samples_per_task=2)
    with pytest.raises(ToolError, match="boom"):
        run(config, backend, reward_fn=good_reward)
    assert backend.save_calls == 0


def test_tool_error_after_finished_step_saves_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-W")]
    _stub_split(monkeypatch, tasks)
    # Step 1 finishes (sample + train); the second train call fails.
    backend = FakeBackend({"TASK-W": 1}, fail_train_on=frozenset({2}))
    config = make_config(tasks_per_step=1, prepass_tasks=1, samples_per_task=2)
    with pytest.raises(ToolError, match="boom"):
        run(config, backend, reward_fn=good_reward)
    assert len(backend.train_calls) == 2
    assert backend.save_calls == 1


def test_reward_tool_error_after_finished_step_saves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-V")]
    _stub_split(monkeypatch, tasks)
    backend = FakeBackend({"TASK-V": 1}, prepass_sample_calls=1)
    config = make_config(tasks_per_step=1, prepass_tasks=1, samples_per_task=2)
    calls = {"count": 0}

    def flaky_reward(answer: str, task_spec: str) -> float:
        calls["count"] += 1
        # Pre-pass (2 calls) and the first step (2 calls) succeed; the second step fails.
        if calls["count"] > 4:
            raise ToolError("boom: reward failed")
        return good_reward(answer, task_spec)

    with pytest.raises(ToolError, match="boom"):
        run(config, backend, reward_fn=flaky_reward)
    assert len(backend.train_calls) == 1
    assert backend.save_calls == 0


def test_default_reward_binds_config_resource_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task("TASK-D")]
    _stub_split(monkeypatch, tasks)
    seen: list[int] = []

    def fake_reward(answer: str, task_spec: str, *, resource_limit: int) -> float:
        seen.append(resource_limit)
        return 0.0

    monkeypatch.setattr(grpo, "reward", fake_reward)
    backend = FakeBackend({"TASK-D": 0})
    config = make_config(prepass_tasks=1, tasks_per_step=1, samples_per_task=2)
    result = run(config, backend)
    assert seen != []
    assert all(limit == Config().dafny.resource_limit for limit in seen)
    assert result.stop_reason == "all groups dropped"


def test_same_seed_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [make_task(f"TASK-{i}") for i in range(4)]
    _stub_split(monkeypatch, tasks)
    config = make_config(tasks_per_step=4, samples_per_task=2, max_spend_usd=0.025)
    first = run(config, FakeBackend({f"TASK-{i}": 1 for i in range(4)}, prepass_sample_calls=4), reward_fn=good_reward)
    second = run(config, FakeBackend({f"TASK-{i}": 1 for i in range(4)}, prepass_sample_calls=4), reward_fn=good_reward)
    assert first == second


def test_types_are_frozen() -> None:
    sample = Sample(text="t", prompt_tokens=1, completion_tokens=2)
    stats = TrainStats(tokens=3)
    config = make_config()
    result = RunResult(steps_done=0, spend_usd=0.0, adapter_id=None, stop_reason="spend cap")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.text = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.tokens = 4  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.steps_done = 1  # type: ignore[misc]


def test_grpo_imports_no_vendor_sdk() -> None:
    source = Path(grpo.__file__).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    assert "fireworks" not in roots
    assert "trl" not in roots


def test_no_dynamic_vendor_import() -> None:
    source = Path(grpo.__file__).read_text(encoding="utf-8")
    assert "__import__" not in source
    assert "importlib" not in source


def test_reward_fn_signature() -> None:
    callback: Callable[[str, str], float] = good_reward
    assert callback("good", "spec") == 1.0


def test_config_defaults_match_ticket() -> None:
    assert GrpoConfig.__dataclass_fields__["max_spend_usd"].default == 50
    assert GrpoConfig.__dataclass_fields__["prepass_tasks"].default == 64
    assert GrpoConfig.__dataclass_fields__["tasks_per_step"].default == 16
    assert GrpoConfig.__dataclass_fields__["samples_per_task"].default == 8
    assert GrpoConfig.__dataclass_fields__["lora_rank"].default == 16
    assert GrpoConfig.__dataclass_fields__["learning_rate"].default == 1e-5
    assert GrpoConfig.__dataclass_fields__["base_adapter"].default == ""
