"""The GRPO training loop (docs/plan.md, Training).

The loop runs against a small ``Backend`` protocol (sampling, training and saving stay behind it,
so the Fireworks backend of a later change plugs in without touching this module) and scores every
attempt with the graded Dafny reward. Offline tests use a fake backend and a fake reward; the real
reward is ``training.reward.reward`` bound to the resource limit of ``pcc.config``.
"""

from __future__ import annotations

import random
import statistics
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pcc.config import Config
from pcc.core import ToolError
from pcc.evaluation import Task, split
from training.prompts import SYSTEM_MESSAGE
from training.reward import reward

_TRAIN_DATASET = "vericoding"
_TRAIN_SPLIT: Literal["train"] = "train"

# The reward that counts as "the model solved the task" (the verified grade of the reward table).
_SOLVED_REWARD = 1.0

# Added to the group standard deviation so a nearly constant group keeps finite advantages.
_ADVANTAGE_EPSILON = 1e-6

# Keys of the Vericoding payload that form the immutable Dafny task prefix (see
# ``pcc.evaluation.datasets``): the supplied specification and method declaration.
_TASK_SPEC_KEYS = ("vc-preamble", "vc-helpers", "vc-spec")


@dataclass(frozen=True, slots=True)
class Sample:
    """One sampled completion with its token counts."""

    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class TrainStats:
    """The token count one ``Backend.train`` call consumed."""

    tokens: int


@runtime_checkable
class Backend(Protocol):
    """Sampling, training and saving behind one door; the Fireworks backend implements this."""

    def sample(self, prompt: str, n: int) -> list[Sample]:
        """Sample ``n`` completions for ``prompt``."""
        ...

    def train(self, batch: list[tuple[str, str, float]]) -> TrainStats:
        """Train one step on ``(prompt, completion, advantage)`` tuples."""
        ...

    def save(self) -> str:
        """Persist the trained adapter; return its id."""
        ...


@dataclass(frozen=True, slots=True)
class GrpoConfig:
    """The GRPO run configuration (``training/configs/glimmer_fireworks.toml``)."""

    model: str  # the base model the run starts from
    prefill_usd_per_1m: float  # USD per 1M prompt tokens
    cached_prefill_usd_per_1m: float  # USD per 1M cached prompt tokens; kept for pricing, unused
    sample_usd_per_1m: float  # USD per 1M sampled tokens
    train_usd_per_1m: float  # USD per 1M train tokens
    max_spend_usd: float = 50.0  # spend cap of the run
    prepass_tasks: int = 64  # training tasks sampled in the pre-pass
    seed: int = 2026  # fixed seed for the pre-pass and step draws
    tasks_per_step: int = 16  # training tasks drawn per step
    samples_per_task: int = 8  # attempts sampled per task
    lora_rank: int = 16  # LoRA rank of the trained adapter
    learning_rate: float = 1e-5  # learning rate of the run
    base_adapter: str = ""  # the SFT 2 adapter the run starts from (set after #80)

    def __post_init__(self) -> None:
        if self.prepass_tasks < 1:
            raise ValueError(f"prepass_tasks must be at least 1, got {self.prepass_tasks}")
        if self.tasks_per_step < 1:
            raise ValueError(f"tasks_per_step must be at least 1, got {self.tasks_per_step}")
        if self.samples_per_task < 1:
            raise ValueError(f"samples_per_task must be at least 1, got {self.samples_per_task}")
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be at least 1, got {self.lora_rank}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_spend_usd < 0:
            raise ValueError(f"max_spend_usd must not be negative, got {self.max_spend_usd}")
        for name in (
            "prefill_usd_per_1m",
            "cached_prefill_usd_per_1m",
            "sample_usd_per_1m",
            "train_usd_per_1m",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative, got {getattr(self, name)}")


@dataclass(frozen=True, slots=True)
class RunResult:
    """The outcome of one ``run``: finished steps, estimated spend and how the run stopped."""

    steps_done: int
    spend_usd: float
    adapter_id: str | None
    stop_reason: str


def load_config(path: str | Path) -> GrpoConfig:
    """Read the TOML file at ``path`` into a ``GrpoConfig``."""
    data = tomllib.loads(Path(path).read_bytes().decode("utf-8"))
    return GrpoConfig(**data)


def estimate_spend(prompt_tokens: int, sample_tokens: int, train_tokens: int, config: GrpoConfig) -> float:
    """Estimate the USD cost of token counts with the rates of ``config``.

    All prompt tokens are charged at the prefill rate; the cached prefill rate stays in the
    config unused. One million prompt, sample and train tokens cost 12.70 USD at the shipped rates.
    """
    million = 1_000_000.0
    return (
        (prompt_tokens / million) * config.prefill_usd_per_1m
        + (sample_tokens / million) * config.sample_usd_per_1m
        + (train_tokens / million) * config.train_usd_per_1m
    )


def _task_spec(task: Task) -> str:
    """The immutable Dafny task prefix of a Vericoding training task.

    This joins the documented ``vc-*`` payload parts (preamble, helpers, spec) and ends at the
    opening method brace. It stands in for the prompt builder of the evaluation until that change
    lands; the evaluation prompt and this prefix are replaced together then.
    """
    parts = [task.payload.get(key, "") for key in _TASK_SPEC_KEYS]
    head = "\n".join(part for part in parts if isinstance(part, str) and part.strip())
    return f"{head}\n{{\n" if head else ""


def _render_prompt(task_spec: str) -> str:
    """Render the training prompt for ``task_spec``.

    This stands in for the evaluation prompt builder until that change lands, so training sees the
    same prompt shape the evaluation sends: the shared system message plus the task prefix.
    """
    return f"{SYSTEM_MESSAGE}\n\n{task_spec}"


def _advantages(rewards: list[float]) -> list[float] | None:
    """Normalize one attempt group to zero-mean advantages, or ``None`` when the group is dropped.

    A group whose rewards are all equal teaches nothing and is dropped. Otherwise each advantage
    is ``(reward - group mean) / (group std + 1e-6)`` with the population standard deviation, so
    the advantages of a kept group sum to 0.
    """
    if len(set(rewards)) <= 1:
        return None
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    return [(reward - mean) / (std + _ADVANTAGE_EPSILON) for reward in rewards]


def run(
    config: GrpoConfig,
    backend: Backend,
    *,
    reward_fn: Callable[[str, str], float] | None = None,
) -> RunResult:
    """Run the GRPO loop described in docs/plan.md (Training, GRPO) and return its result.

    The pre-pass draws ``config.prepass_tasks`` training tasks with the fixed seed, samples each
    one ``config.samples_per_task`` times and keeps the tasks the starting model solves at least
    once but not always; its cost counts against ``config.max_spend_usd``. Each step draws
    ``config.tasks_per_step`` tasks from the kept set, samples attempts, scores them with
    ``reward_fn`` (default: the graded Dafny reward bound to the resource limit of
    ``pcc.config``), drops the groups whose rewards are all equal and trains on the rest.

    A step starts only if the spend so far plus the previous step's cost stays within
    ``config.max_spend_usd``; before the first step the estimate is the pre-pass cost per sample
    times ``tasks_per_step * samples_per_task``. When the rule stops the run, ``save()`` is
    called and ``stop_reason`` is "spend cap". When every group is dropped, the run stops with
    "all groups dropped" and calls no ``train``. ``save()`` runs once on every normal stop,
    except when the pre-pass alone exceeded the cap (then no step ran and ``save()`` is skipped).
    A ``ToolError`` from the backend propagates, after ``save()`` when at least one step finished.
    """
    if reward_fn is None:
        resource_limit = Config().dafny.resource_limit

        def _default_reward(answer: str, task_spec: str) -> float:
            return reward(answer, task_spec, resource_limit=resource_limit)

        reward_fn = _default_reward

    rng = random.Random(config.seed)
    pool = list(split(_TRAIN_DATASET, _TRAIN_SPLIT))
    prepass = rng.sample(pool, min(config.prepass_tasks, len(pool)))

    spend = 0.0
    prepass_samples = 0
    kept: list[Task] = []
    for task in prepass:
        task_spec = _task_spec(task)
        prompt = _render_prompt(task_spec)
        samples = backend.sample(prompt, config.samples_per_task)
        prepass_samples += len(samples)
        spend += estimate_spend(
            sum(sample.prompt_tokens for sample in samples),
            sum(sample.completion_tokens for sample in samples),
            0,
            config,
        )
        solved = sum(1 for sample in samples if reward_fn(sample.text, task_spec) == _SOLVED_REWARD)
        if 1 <= solved <= config.samples_per_task - 1:
            kept.append(task)

    if spend > config.max_spend_usd:
        return RunResult(steps_done=0, spend_usd=spend, adapter_id=None, stop_reason="spend cap")
    if not kept:
        return RunResult(steps_done=0, spend_usd=spend, adapter_id=backend.save(), stop_reason="all groups dropped")

    estimate = (spend / prepass_samples if prepass_samples else 0.0) * config.tasks_per_step * config.samples_per_task
    steps_done = 0
    stop_reason = "spend cap"
    try:
        while True:
            if spend + estimate > config.max_spend_usd:
                stop_reason = "spend cap"
                break
            batch: list[tuple[str, str, float]] = []
            step_prompt_tokens = 0
            step_sample_tokens = 0
            for task in rng.sample(kept, min(config.tasks_per_step, len(kept))):
                task_spec = _task_spec(task)
                prompt = _render_prompt(task_spec)
                samples = backend.sample(prompt, config.samples_per_task)
                step_prompt_tokens += sum(sample.prompt_tokens for sample in samples)
                step_sample_tokens += sum(sample.completion_tokens for sample in samples)
                rewards = [reward_fn(sample.text, task_spec) for sample in samples]
                advantages = _advantages(rewards)
                if advantages is None:
                    continue
                batch.extend(
                    (prompt, sample.text, advantage) for sample, advantage in zip(samples, advantages, strict=True)
                )
            if not batch:
                spend += estimate_spend(step_prompt_tokens, step_sample_tokens, 0, config)
                stop_reason = "all groups dropped"
                break
            stats = backend.train(batch)
            estimate = estimate_spend(step_prompt_tokens, step_sample_tokens, stats.tokens, config)
            spend += estimate
            steps_done += 1
    except ToolError:
        if steps_done >= 1:
            backend.save()
        raise
    return RunResult(steps_done=steps_done, spend_usd=spend, adapter_id=backend.save(), stop_reason=stop_reason)


__all__ = [
    "Backend",
    "GrpoConfig",
    "RunResult",
    "Sample",
    "TrainStats",
    "estimate_spend",
    "load_config",
    "run",
]
