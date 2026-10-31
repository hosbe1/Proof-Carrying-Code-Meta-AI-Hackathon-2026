"""Export verified training solutions to Fireworks SFT chat format (docs/plan.md, Training).

Each sampling run directory holds ``samples.jsonl`` (every attempt with its verdict) and
optionally ``proofs.jsonl`` (proof-only lines with the prompt of the last ``proof`` call).
``python -m training.export_sft runs/<name>... --out sft.jsonl [--include <jsonl>...]`` writes
one chat line per verified, deduplicated solution. ``--include`` merges lines that already have
the chat format (the round 0 output). Only train-split tasks are used: a task ID from a test
split, or a user message equal to the prompt of a test-split task, raises ``ValueError``.

``build_prompt`` is the reference Vericoding prompt used both to rebuild the user message of a
run solution and to guard the test split; the harness and the round 0 tool use this same
function so prompts compare byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from pcc.evaluation import datasets
from pcc.evaluation.datasets import Task
from training.prompts import SYSTEM_MESSAGE

SAMPLES_FILENAME = "samples.jsonl"
PROOFS_FILENAME = "proofs.jsonl"

_ROLES = ("system", "user", "assistant")

_PROMPT_INSTRUCTIONS = "Complete the Dafny method below so that it verifies. Return only the complete Dafny program."


class ChatMessage(TypedDict):
    """One Fireworks chat turn."""

    role: str
    content: str


class ChatLine(TypedDict):
    """One SFT training line in the Fireworks chat format."""

    messages: list[ChatMessage]


@dataclass(frozen=True, slots=True)
class TaskIndex:
    """Train-split tasks by ID, plus the test-split guards: test IDs and their prompts."""

    tasks: Mapping[str, Task]
    test_ids: frozenset[str]
    test_prompts: frozenset[str]


def build_prompt(task: Task) -> str:
    """The Vericoding user prompt for ``task``: instructions plus spec, never the answer.

    For a Vericoding row this joins the description, preamble, helpers, spec and postamble,
    skipping blank parts; ``vc-code`` holds the reference answer, so it stays out. For a
    HumanEvalFix row it is the instructions plus the ``prompt`` stub. Anything else raises
    ``ValueError`` instead of guessing a prompt.
    """
    payload = task.payload
    spec = payload.get("vc-spec")
    if isinstance(spec, str) and spec.strip():
        blocks = [
            payload.get("vc-description"),
            payload.get("vc-preamble"),
            payload.get("vc-helpers"),
            spec,
            payload.get("vc-postamble"),
        ]
        body = "\n\n".join(block for block in blocks if isinstance(block, str) and block.strip())
        if body:
            return f"{_PROMPT_INSTRUCTIONS}\n\n{body}"
        return _PROMPT_INSTRUCTIONS
    stub = payload.get("prompt")
    if isinstance(stub, str) and stub.strip():
        return f"{_PROMPT_INSTRUCTIONS}\n\n{stub}"
    raise ValueError(f"cannot build a training prompt for task {task.id!r}: unsupported payload")


def _records(path: Path) -> Iterator[tuple[int, Any]]:
    """The ``(line number, parsed JSON)`` pairs of ``path``, skipping blank lines.

    Every line must hold a JSON object; anything else raises ``ValueError`` naming the file
    and the line, because silently dropping training data hides contamination.
    """
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record: Any = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path} line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path} line {lineno}: expected a JSON object")
        yield lineno, record


def _checked_task_id(record: Any, label: str, *, index: TaskIndex) -> str:
    """The task ID of ``record``; test-split, non-train and unknown IDs raise ``ValueError``.

    The test check runs before anything else, so even an unverified line carrying a test ID
    fails the export: test data must never reach training, not even by accident. ``index.tasks``
    holds only the train split, so any other ID (unknown, or excluded from train through
    another dataset's test ``source_task``) fails as a non-train ID.
    """
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f'{label}: expected a non-empty "task_id" string')
    if task_id in index.test_ids:
        raise ValueError(f"{label}: task {task_id!r} is on the test split and must not enter training")
    if task_id not in index.tasks:
        raise ValueError(f"{label}: unknown task ID {task_id!r}: not on the train split, must not enter training")
    return task_id


def _checked_messages(record: Any, label: str) -> list[ChatMessage]:
    """The prompt messages of a ``proofs.jsonl`` line: non-empty, roles and strings only.

    The stored prompt must hold no assistant turn (it is the prompt of one ``proof`` call) and
    at least one user turn, whose content the test-split guard and the dedup key use.
    """
    raw = record.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f'{label}: expected a non-empty "messages" list')
    messages: list[ChatMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{label}: expected a JSON object for every message")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or role not in _ROLES or not isinstance(content, str):
            raise ValueError(f"{label}: every message needs a role and a string content")
        messages.append({"role": role, "content": content})
    if any(message["role"] == "assistant" for message in messages):
        raise ValueError(f"{label}: the prompt must not hold an assistant message")
    if not any(message["role"] == "user" for message in messages):
        raise ValueError(f"{label}: the prompt holds no user message")
    return messages


def _user_content(messages: Sequence[ChatMessage]) -> str:
    """The content of the first user turn of ``messages``."""
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    raise ValueError("chat line holds no user message")


def _iter_run_samples(run_dir: Path, *, index: TaskIndex) -> Iterator[tuple[str, str, list[ChatMessage]]]:
    """The ``(user, answer, messages)`` triples of the verified ``samples.jsonl`` lines."""
    path = run_dir / SAMPLES_FILENAME
    if not path.is_file():
        return
    for lineno, record in _records(path):
        label = f"{path} line {lineno}"
        task_id = _checked_task_id(record, label, index=index)
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f'{label}: expected a non-empty "answer" string')
        verified = record.get("verified", False)
        if not isinstance(verified, bool):
            raise ValueError(f'{label}: expected a "verified" boolean')
        if not verified:
            continue
        user = build_prompt(index.tasks[task_id])
        messages: list[ChatMessage] = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
        yield user, answer, messages


def _iter_run_proofs(run_dir: Path, *, index: TaskIndex) -> Iterator[tuple[str, str, list[ChatMessage]]]:
    """The ``(user, answer, messages)`` triples of the run's ``proofs.jsonl`` lines, if any."""
    path = run_dir / PROOFS_FILENAME
    if not path.is_file():
        return
    for lineno, record in _records(path):
        label = f"{path} line {lineno}"
        _checked_task_id(record, label, index=index)
        prompt = _checked_messages(record, label)
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f'{label}: expected a non-empty "answer" string')
        assistant_turn: ChatMessage = {"role": "assistant", "content": answer}
        messages: list[ChatMessage] = [*prompt, assistant_turn]
        yield _user_content(prompt), answer, messages


def _iter_include(path: Path) -> Iterator[tuple[str, str, list[ChatMessage]]]:
    """The ``(user, answer, messages)`` triples of a file that already has the chat format."""
    for lineno, record in _records(path):
        label = f"{path} line {lineno}"
        raw = record.get("messages")
        if not isinstance(raw, list) or not raw:
            raise ValueError(f'{label}: expected a non-empty "messages" list')
        messages: list[ChatMessage] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(f"{label}: expected a JSON object for every message")
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or role not in _ROLES or not isinstance(content, str):
                raise ValueError(f"{label}: every message needs a role and a string content")
            messages.append({"role": role, "content": content})
        if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
            raise ValueError(f"{label}: a chat line must end with a non-empty assistant message")
        if not any(message["role"] == "user" for message in messages):
            raise ValueError(f"{label}: a chat line holds no user message")
        yield _user_content(messages), messages[-1]["content"], messages


def collect_lines(run_dirs: Sequence[Path], include_paths: Sequence[Path], *, index: TaskIndex) -> list[ChatLine]:
    """Every verified, deduplicated chat line of the runs and the ``--include`` files.

    Inputs are processed in order (each run's samples, then its proofs, then the includes) and
    deduplicated together by ``(user, answer)``: the first occurrence wins. A user message equal
    to the prompt of a test-split task raises ``ValueError``. A run that yields no candidate
    line warns instead of failing, so an empty run still yields an (empty) output file.
    """
    seen: set[tuple[str, str]] = set()
    lines: list[ChatLine] = []

    def append(user: str, answer: str, messages: list[ChatMessage], label: str) -> None:
        if user in index.test_prompts:
            raise ValueError(f"{label}: user message equals the prompt of a task on the test split")
        if (user, answer) not in seen:
            seen.add((user, answer))
            lines.append({"messages": messages})

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory {run_dir} does not exist")
        yielded = 0
        for user, answer, messages in _iter_run_samples(run_dir, index=index):
            yielded += 1
            append(user, answer, messages, f"{run_dir / SAMPLES_FILENAME}")
        for user, answer, messages in _iter_run_proofs(run_dir, index=index):
            yielded += 1
            append(user, answer, messages, f"{run_dir / PROOFS_FILENAME}")
        if yielded == 0:
            warnings.warn(f"run {run_dir} contributed no training lines", UserWarning, stacklevel=2)
    for path in include_paths:
        if not path.is_file():
            raise FileNotFoundError(f"include file {path} does not exist")
        for user, answer, messages in _iter_include(path):
            append(user, answer, messages, str(path))
    return lines


def build_index(cache_dir: Path | None = None, splits_dir: Path | None = None) -> TaskIndex:
    """Train-split tasks of every dataset with the test-split guards over every test split.

    ``tasks`` holds only the train split (via ``datasets.split(name, "train")``), so a benchmark
    task excluded from train through another dataset's test ``source_task`` is rejected as a
    non-train ID instead of leaking in. Both the task-ID check and the prompt check still cover
    every test split.
    """
    train_tasks: dict[str, Task] = {}
    test_tasks: dict[str, Task] = {}
    test_ids: set[str] = set()
    for name in (datasets.VERICODING, datasets.HUMANEVALFIX):
        for task in datasets.split(name, "train", cache_dir, splits_dir):
            train_tasks[task.id] = task
        for task in datasets.split(name, "test", cache_dir, splits_dir):
            test_tasks[task.id] = task
            test_ids.add(task.id)
    return TaskIndex(
        tasks=train_tasks,
        test_ids=frozenset(test_ids),
        test_prompts=frozenset(build_prompt(task) for task in test_tasks.values()),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    cache_dir: Path | None = None,
    splits_dir: Path | None = None,
) -> int:
    """Entry point for ``python -m training.export_sft``; returns the process exit code.

    ``cache_dir`` and ``splits_dir`` override the dataset cache and the split files (the tests
    use them for fixtures); both default to the committed locations. ``ValueError`` from a
    test-split guard propagates, so contamination fails the command loudly.
    """
    parser = argparse.ArgumentParser(description="Export verified training solutions to Fireworks SFT chat format.")
    parser.add_argument("runs", nargs="+", type=Path, help="sampling run directories to export")
    parser.add_argument("--out", required=True, type=Path, help="output JSONL file in chat format")
    parser.add_argument(
        "--include",
        nargs="*",
        type=Path,
        default=[],
        help="extra files that already have the chat format",
    )
    args = parser.parse_args(argv)
    lines = collect_lines(args.runs, args.include, index=build_index(cache_dir, splits_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "PROOFS_FILENAME",
    "SAMPLES_FILENAME",
    "ChatLine",
    "ChatMessage",
    "TaskIndex",
    "build_index",
    "build_prompt",
    "collect_lines",
    "main",
]
