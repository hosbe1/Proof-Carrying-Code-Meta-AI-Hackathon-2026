"""Rejection sampling for the proof step (docs/plan.md, Training).

The model proposes several attempts per Vericoding training task, Dafny (through
``training.reward.reward``) keeps the verified ones, and they become SFT data in the
chat format of ``training/export_sft.py``. Round 0 samples the contributor model through
the Meta endpoint; round 1 samples the trained model on the still-unsolved tasks through
the Glimmer deployment. Attempts are free text, so no schema is sent.

``run`` drives one round; ``python -m training.rejection`` is its CLI. Sampling and
verification are two phases: attempts are sampled concurrently (every raw attempt is
appended to ``<out>.attempts.jsonl`` under a lock), then ``reward_fn`` scores every
attempt of the finished tasks in a process pool. An attempt is verified only when the
reward is exactly 1.0; 0 and -1 are not verified.

Files next to ``<out>``: ``<out>.attempts.jsonl`` (every raw attempt ever sampled, with
``task_id``, ``prompt_sha256``, ``prompt`` and ``answer``), ``<out>.tasks.jsonl`` (one
line per finished task with ``prompt_sha256``, ``attempts`` and ``solved``) and
``<out>.summary.json`` (this invocation's ``RoundSummary``). ``<out>`` itself is
rewritten from the verified attempts at the end of each run: per task, distinct answers
in first-occurrence order, at most ``max_per_task`` lines.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcc.config import Config
from pcc.core import ToolError, Usage
from pcc.evaluation.datasets import Task, split
from pcc.evaluation.vericoding import task_spec_of
from pcc.models import BudgetedClient, BudgetExceeded, Message, ModelClient, OpenAICompatClient
from pcc.models.router import build_client
from training.prompts import SYSTEM_MESSAGE, build_prompt
from training.reward import reward

#: The step name every model call of this tool is recorded under.
_STEP = "proof"

#: Stop reasons that exit 0: the round finished, or the spend cap stopped sampling.
_OK_STOP_REASONS = ("done", "cost cap")


@dataclass(frozen=True, slots=True)
class RoundSummary:
    """The outcome of one ``run`` invocation: what this invocation solved, ran and spent."""

    tasks_solved: int  # finished tasks with at least one verified attempt
    tasks_run: int  # tasks sampled to completion, verified and recorded
    attempts: int  # model calls made by this invocation
    input_tokens: int  # input tokens reported by this invocation's calls
    output_tokens: int  # output tokens reported by this invocation's calls
    cost_usd: float  # cost reported by this invocation's calls
    wall_seconds: float  # wall-clock seconds of this invocation
    stop_reason: str  # "done", "cost cap" or "endpoint: <message>"

    def __str__(self) -> str:
        return (
            f"{self.tasks_solved} of {self.tasks_run} tasks solved, "
            f"{self.attempts} attempts, "
            f"{self.input_tokens} input and {self.output_tokens} output tokens, "
            f"cost ${self.cost_usd:.4f}, {self.wall_seconds:.1f}s ({self.stop_reason})"
        )


@dataclass(frozen=True, slots=True)
class _AttemptRecord:
    """One sampled attempt with the usage its model call reported."""

    task_index: int
    prompt_sha256: str
    prompt: str
    answer: str
    usage: Usage


def prompt_sha256(task: Task) -> str:
    """Lowercase hex SHA-256 of the exact UTF-8 bytes of ``build_prompt(task)``."""
    return hashlib.sha256(build_prompt(task).encode("utf-8")).hexdigest()


def _default_reward(answer: str, task_spec: str) -> float:
    """The graded Dafny reward bound to the configured resource limit.

    Module-level so the process pool can pickle it; ``training.grpo`` binds the same
    reward the same way.
    """
    return reward(answer, task_spec, resource_limit=Config().dafny.resource_limit)


def _chat_line(prompt: str, answer: str) -> dict[str, Any]:
    """One SFT chat line in the format of ``training/export_sft.py``."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    """The parsed JSON objects of ``path``; a broken resume file is a ``ToolError``."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot resume from {label} {path}: {exc}") from exc
    try:
        lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except ValueError as exc:
        raise ToolError(f"cannot resume from {label} {path}: {exc}") from exc
    for lineno, line in enumerate(lines, start=1):
        if not isinstance(line, dict):
            raise ToolError(f"cannot resume from {label} {path}: line {lineno} is not a JSON object")
    return lines


def _load_skip_prompts(path: str | Path | None) -> set[str]:
    """The user prompts occurring in a ``--skip-solved`` chat file (lenient: skips bad lines)."""
    if path is None:
        return set()
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot read --skip-solved {path}: {exc}") from exc
    prompts: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            record: Any = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
                prompts.add(message["content"])
    return prompts


def _load_finished(path: Path) -> set[str]:
    """The ``prompt_sha256`` values recorded as finished in a tasks file."""
    if not path.is_file():
        return set()
    finished: set[str] = set()
    for lineno, record in enumerate(_read_jsonl_objects(path, "tasks"), start=1):
        sha = record.get("prompt_sha256")
        if not isinstance(sha, str) or not sha:
            raise ToolError(f"cannot resume from tasks {path}: line {lineno} has no prompt_sha256")
        finished.add(sha)
    return finished


def _load_retained(path: Path) -> dict[str, list[tuple[str, str]]]:
    """The retained ``(prompt, answer)`` attempts per ``prompt_sha256``, in file order."""
    retained: dict[str, list[tuple[str, str]]] = {}
    if not path.is_file():
        return retained
    for lineno, record in enumerate(_read_jsonl_objects(path, "attempts"), start=1):
        sha = record.get("prompt_sha256")
        prompt = record.get("prompt")
        answer = record.get("answer")
        if not isinstance(sha, str) or not sha or not isinstance(prompt, str) or not isinstance(answer, str):
            raise ToolError(f"cannot resume from attempts {path}: line {lineno} is not an attempt")
        retained.setdefault(sha, []).append((prompt, answer))
    return retained


def _split_chat_line(line: dict[str, Any], path: Path) -> tuple[str, str]:
    """The ``(user, assistant)`` contents of a stored chat line."""
    messages = line.get("messages")
    if not isinstance(messages, list):
        raise ToolError(f"cannot resume from output {path}: a line has no messages list")
    user: str | None = None
    assistant: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            raise ToolError(f"cannot resume from output {path}: a message is not an object")
        if message.get("role") == "user" and isinstance(message.get("content"), str) and user is None:
            user = message["content"]
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            assistant = message["content"]
    if user is None or assistant is None:
        raise ToolError(f"cannot resume from output {path}: a line has no user or assistant message")
    return user, assistant


def run(
    tasks: Sequence[Task],
    client: ModelClient,
    *,
    model: str,
    attempts: int,
    out: str | Path,
    reward_fn: Callable[[str, str], float] | None = None,
    skip_solved: str | Path | None = None,
    concurrency: int = 64,
    workers: int | None = None,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    max_per_task: int = 2,
    max_cost_usd: float | None = None,
    resume: bool = False,
) -> RoundSummary:
    """Sample ``attempts`` proofs per task, keep the verified ones, and return the summary.

    Every attempt is one ``complete`` call with the shared system message and
    ``build_prompt(task)`` as the user message (``step="proof"``) and is appended to
    ``<out>.attempts.jsonl`` as it is sampled. After sampling, ``reward_fn`` (default:
    the graded Dafny reward bound to ``Config().dafny.resource_limit``) scores every
    attempt of the finished tasks in a process pool; the reward sees
    ``(answer, task_spec_of(task))``. A ``ToolError`` from ``reward_fn`` propagates.

    ``skip_solved`` skips tasks whose exact user prompt occurs in that chat file;
    ``resume`` additionally skips the tasks recorded as finished in
    ``<out>.tasks.jsonl`` (the union: skipped by both means no new call). For a task
    with retained attempts but no finished-task record, only the missing attempts up to
    ``attempts`` are sampled, then all of them are verified. ``<out>`` is rebuilt from
    the verified attempts, retaining resumed lines; ``<out>.summary.json`` and the
    returned summary describe this invocation only.

    With ``max_cost_usd`` the client is wrapped in
    ``BudgetedClient(client, max_cost_usd=cap, max_calls=sys.maxsize,
    max_seconds=float("inf"))`` so only the dollar cap constrains the run. The call
    that crosses the cap completes; the next call raises ``BudgetExceeded``, which stops
    sampling with ``stop_reason`` ``"cost cap"``. A ``ToolError`` from the endpoint
    stops sampling with ``stop_reason`` ``"endpoint: <message>"``; in both cases the
    attempts sampled so far are kept and the tasks that finished sampling are verified
    and recorded. An existing ``out`` without ``resume`` raises ``ValueError``.
    """
    start = time.monotonic()
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if max_per_task < 1:
        raise ValueError(f"max_per_task must be at least 1, got {max_per_task}")
    out_path = Path(out)
    if out_path.exists() and not resume:
        raise ValueError(f"output {out_path} exists; pass resume=True to continue the run")
    attempts_path = Path(str(out_path) + ".attempts.jsonl")
    tasks_path = Path(str(out_path) + ".tasks.jsonl")
    summary_path = Path(str(out_path) + ".summary.json")

    check_fn = reward_fn if reward_fn is not None else _default_reward
    prompts = [build_prompt(task) for task in tasks]
    shas = [prompt_sha256(task) for task in tasks]
    skipped = _load_skip_prompts(skip_solved)
    finished = _load_finished(tasks_path) if resume else set()
    retained = _load_retained(attempts_path) if resume else {}
    prev_lines = _read_jsonl_objects(out_path, "output") if resume and out_path.is_file() else []
    if resume:
        attempts_path.touch(exist_ok=True)
    else:
        attempts_path.write_text("", encoding="utf-8")

    if len(tasks) == 0:
        warnings.warn("no tasks to sample; writing an empty output file", UserWarning, stacklevel=2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        tasks_path.write_text("", encoding="utf-8")
        summary = RoundSummary(0, 0, 0, 0, 0, 0.0, time.monotonic() - start, "done")
        summary_path.write_text(json.dumps(dataclasses.asdict(summary), sort_keys=True) + "\n", encoding="utf-8")
        print(summary)
        return summary

    order = [index for index in range(len(tasks)) if shas[index] not in finished and prompts[index] not in skipped]
    needed = {index: max(0, attempts - len(retained.get(shas[index], []))) for index in order}

    active: ModelClient = client
    if max_cost_usd is not None:
        active = BudgetedClient(client, max_cost_usd=max_cost_usd, max_calls=sys.maxsize, max_seconds=float("inf"))

    new_records: dict[int, list[_AttemptRecord]] = {index: [] for index in order}
    lock = threading.Lock()

    def _sample_one(index: int) -> _AttemptRecord:
        completion = active.complete(
            model,
            (Message("system", SYSTEM_MESSAGE), Message("user", prompts[index])),
            step=_STEP,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        record = _AttemptRecord(index, shas[index], prompts[index], completion.text, completion.usage)
        line = json.dumps(
            {
                "task_id": tasks[index].id,
                "prompt_sha256": shas[index],
                "prompt": prompts[index],
                "answer": completion.text,
            },
            ensure_ascii=False,
        )
        with lock, attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return record

    stop: str | None = None
    with ThreadPoolExecutor(max_workers=concurrency) as sampler:
        futures: list[tuple[int, Future[_AttemptRecord]]] = []
        for index in order:
            for _ in range(needed[index]):
                futures.append((index, sampler.submit(_sample_one, index)))
        for index, future in futures:
            if stop is not None:
                future.cancel()
            try:
                new_records[index].append(future.result())
            except BudgetExceeded:
                if stop is None:
                    stop = "cost cap"
            except ToolError as exc:
                if stop is None:
                    stop = f"endpoint: {exc}"
            except CancelledError:
                continue

    finished_order = [
        index for index in order if len(retained.get(shas[index], [])) + len(new_records[index]) >= attempts
    ]
    verify_jobs: list[tuple[int, str, str]] = []
    for index in finished_order:
        frozen = task_spec_of(tasks[index])
        for _, answer in retained.get(shas[index], []):
            verify_jobs.append((index, answer, frozen))
        for record in new_records[index]:
            verify_jobs.append((index, record.answer, frozen))
    verified: dict[int, list[str]] = {index: [] for index in finished_order}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scored = [pool.submit(check_fn, answer, frozen) for _, answer, frozen in verify_jobs]
        for (index, answer, _), scored_future in zip(verify_jobs, scored, strict=True):
            if scored_future.result() == 1.0:
                verified[index].append(answer)

    prev_by_prompt: dict[str, list[str]] = {}
    prev_line_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for line in prev_lines:
        user, assistant = _split_chat_line(line, out_path)
        prev_by_prompt.setdefault(user, []).append(assistant)
        prev_line_by_prompt.setdefault(user, []).append(line)

    new_line_by_prompt: dict[str, list[dict[str, Any]]] = {}
    solved: dict[int, bool] = {}
    for index in finished_order:
        prompt = prompts[index]
        seen = set(prev_by_prompt.get(prompt, []))
        distinct: list[str] = []
        for answer in verified[index]:
            if answer not in seen:
                seen.add(answer)
                distinct.append(answer)
        allowance = max(0, max_per_task - len(prev_by_prompt.get(prompt, [])))
        new_line_by_prompt[prompt] = [_chat_line(prompt, answer) for answer in distinct[:allowance]]
        solved[index] = len(verified[index]) > 0

    merged: list[dict[str, Any]] = []
    emitted: set[tuple[str, str]] = set()
    current_prompts = {prompts[index] for index in range(len(tasks))}

    def _emit(line: dict[str, Any]) -> None:
        user, assistant = _split_chat_line(line, out_path)
        if (user, assistant) not in emitted:
            emitted.add((user, assistant))
            merged.append(line)

    for index in range(len(tasks)):
        for line in prev_line_by_prompt.get(prompts[index], []):
            _emit(line)
        for line in new_line_by_prompt.get(prompts[index], []):
            _emit(line)
    for line in prev_lines:
        user, _ = _split_chat_line(line, out_path)
        if user not in current_prompts:
            _emit(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for line in merged:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    with tasks_path.open("a" if resume else "w", encoding="utf-8") as handle:
        for index in finished_order:
            total = len(retained.get(shas[index], [])) + len(new_records[index])
            handle.write(
                json.dumps(
                    {"prompt_sha256": shas[index], "attempts": total, "solved": solved[index]},
                    sort_keys=True,
                )
                + "\n"
            )

    calls = [record for records in new_records.values() for record in records]
    summary = RoundSummary(
        tasks_solved=sum(1 for index in finished_order if solved[index]),
        tasks_run=len(finished_order),
        attempts=len(calls),
        input_tokens=sum(record.usage.input_tokens for record in calls),
        output_tokens=sum(record.usage.output_tokens for record in calls),
        cost_usd=sum(record.usage.cost_usd for record in calls),
        wall_seconds=time.monotonic() - start,
        stop_reason=stop or "done",
    )
    summary_path.write_text(json.dumps(dataclasses.asdict(summary), sort_keys=True) + "\n", encoding="utf-8")
    print(summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """The CLI parser; every option of the ticket is a named argument."""
    parser = argparse.ArgumentParser(
        description="Sample proofs per Vericoding training task and keep the Dafny-verified ones as SFT data."
    )
    parser.add_argument("--endpoint", choices=("meta", "glimmer"), required=True, help="which endpoint to sample")
    parser.add_argument("--model", required=True, help="the model id to send with every call")
    parser.add_argument("--tasks", required=True, help='the task split to sample; only "train" is supported')
    parser.add_argument("--attempts", type=int, default=8, help="attempts sampled per task")
    parser.add_argument("--out", type=Path, required=True, help="output JSONL file in SFT chat format")
    parser.add_argument("--skip-solved", type=Path, default=None, help="chat file whose solved prompts are skipped")
    parser.add_argument("--concurrency", type=int, default=64, help="tasks sampled at once")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processes verifying attempts")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature of every call")
    parser.add_argument("--max-tokens", type=int, default=4096, help="sampling output limit of every call")
    parser.add_argument("--max-per-task", type=int, default=2, help="verified answers written per task")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="spend cap of this run (USD)")
    parser.add_argument("--resume", action="store_true", help="continue into existing output files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m training.rejection``; returns the process exit code.

    Usage errors exit 2; a stopped endpoint exits 2 while a spend cap exits 0. A
    ``ToolError`` from the reward passes through.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.tasks != "train":
        parser.error(f'unknown --tasks {args.tasks!r}; expected "train"')
    if args.attempts < 1:
        parser.error(f"--attempts must be at least 1, got {args.attempts}")
    if args.concurrency < 1:
        parser.error(f"--concurrency must be at least 1, got {args.concurrency}")
    if args.workers is not None and args.workers < 1:
        parser.error(f"--workers must be at least 1, got {args.workers}")
    if args.max_per_task < 1:
        parser.error(f"--max-per-task must be at least 1, got {args.max_per_task}")
    if args.endpoint == "meta" and args.max_cost_usd is None:
        parser.error("--max-cost-usd is required with --endpoint meta")
    if args.out.exists() and not args.resume:
        parser.error(f"--out {args.out} exists; pass --resume to continue the run")

    env = os.environ
    if args.endpoint == "meta":
        client: ModelClient = build_client(Config(), env)
    else:
        price = Config().models.endpoint_prices.get("glimmer", (0.0, 0.0))
        client = OpenAICompatClient(
            env.get("PCC_GLIMMER_BASE_URL", ""),
            env.get("PCC_GLIMMER_API_KEY", ""),
            prices={args.model: price},
        )
    try:
        summary = run(
            split("vericoding", "train"),
            client,
            model=args.model,
            attempts=args.attempts,
            out=args.out,
            skip_solved=args.skip_solved,
            concurrency=args.concurrency,
            workers=args.workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_per_task=args.max_per_task,
            max_cost_usd=args.max_cost_usd,
            resume=args.resume,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if summary.stop_reason in _OK_STOP_REASONS else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["RoundSummary", "main", "prompt_sha256", "run"]
