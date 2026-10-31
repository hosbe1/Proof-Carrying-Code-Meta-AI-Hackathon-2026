"""The Vericoding harness: sample proofs for a spec and verify them with Dafny (issue #46).

Each task sends two messages, the shared ``SYSTEM_MESSAGE`` as the system turn and the
``build_prompt(task)`` output as the user turn, byte for byte, so evaluation and training see
the same prompt. Sampling runs in a thread pool (``--concurrency``); every answer goes through
``anti_cheat`` against the frozen prefix ``task_spec_of(task)`` and then ``verifier.verify`` in a
process pool. Cost comes from completed model usage; once it reaches ``--max-cost-usd`` no new
task is submitted, work already submitted finishes and its actual cost is recorded.

``runs/<name>`` holds ``results.jsonl`` (one line per completed task, in stable task-ID order),
``run.json`` (the protocol fields) and ``summary.md`` (proof rate, cost and time per task).
Tasks whose prompt cannot be built (``build_prompt`` raises ``ValueError``) are skipped before
any sampling and reported in ``skipped.json``; ``run.json`` and ``results.jsonl`` count only
eligible tasks. Rerunning into the same directory skips the task IDs already in
``results.jsonl``. A ``ToolError`` (model unavailable, Dafny missing or of the wrong version)
stops the run; the lines written so far are kept.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pcc import models
from pcc.config import Config, ProverStage
from pcc.core import ToolError
from pcc.evaluation import datasets
from pcc.evaluation.datasets import Task
from pcc.evaluation.prompts import SYSTEM_MESSAGE, build_prompt
from pcc.evaluation.proof_only import build_proof_only_prompt
from pcc.models import Message, ModelClient
from pcc.models.router import ModelUnavailable
from pcc.verifier import anti_cheat, same_executable, verify
from pcc.verifier.ghost import declared_lemma_names
from pcc.verifier.header import header_violations

HARNESS = "vericoding"

#: The committed split file whose SHA-256 is recorded as ``split_sha256`` (it also defines the train complement).
SPLIT_FILE = Path(__file__).resolve().parent / "splits" / "vericoding_test.txt"

#: Keys of the Vericoding payload forming the immutable Dafny task prefix (the same contract as ``training.grpo``).
_PREFIX_KEYS = ("vc-preamble", "vc-helpers", "vc-spec")

#: The model ID that marks a run as the contributor variant.
_CONTRIBUTOR_MODEL = "muse-spark-1.3-contributor"

#: The step name every model call of this harness is recorded under.
_STEP = "vericoding"

#: Sampling options from the evaluation protocol: independent samples, at most 4096 output tokens.
_TEMPERATURE = 0.8
_MAX_TOKENS = 4096

#: The exact ``summary.md`` line marking a run stopped by the spend cap.
CAP_LINE = "stopped at cost cap"


@dataclass(frozen=True, slots=True)
class _Attempt:
    """One sampled answer with the cost and time its model call reported."""

    text: str
    cost_usd: float
    seconds: float


def task_spec_of(task: Task) -> str:
    """The immutable Dafny task prefix of a Vericoding task.

    This joins the nonblank ``vc-preamble``, ``vc-helpers`` and ``vc-spec`` payload parts in
    that order with ``"\\n"`` and appends ``"\\n{\\n"``: the supplied specification and method
    declaration through the opening method brace. ``vc-description`` and ``vc-postamble`` belong
    in the user prompt, not the frozen prefix; ``vc-code`` is the reference answer and never
    enters the prompt. The answer must preserve these bytes, and ``anti_cheat`` checks that.

    A proof-only task (payload ``{"program": ...}``) has no Vericoding prefix; its program
    is the reference the answer is checked against.
    """
    program = task.payload.get("program")
    if isinstance(program, str):
        return program
    parts = [task.payload.get(key, "") for key in _PREFIX_KEYS]
    head = "\n".join(part for part in parts if isinstance(part, str) and part.strip())
    return f"{head}\n{{\n" if head else ""


def _is_proof_only(task: Task) -> bool:
    """Whether ``task`` comes from a proof-only tasks file (payload ``{"program": ...}``)."""
    return isinstance(task.payload.get("program"), str)


def _is_cheat(answer: str, frozen_prefix: str) -> bool:
    """Whether ``answer`` fails the frozen-prefix contract or the cheat scans."""
    if not frozen_prefix or frozen_prefix not in answer:
        return True
    if anti_cheat(answer, frozen_prefix):
        return True
    return bool(header_violations(answer, frozen_prefix))


def _is_proof_only_cheat(answer: str, program: str) -> bool:
    """Whether a proof-only ``answer`` fails the ghost-only contract against ``program``.

    The answer must differ from the program only in ghost code, keep the method
    header and pass the cheat scan; ``lemma_names`` are the lemmas declared in
    the answer itself.
    """
    names = declared_lemma_names(answer)
    if not same_executable(program, answer, lemma_names=names):
        return True
    if header_violations(answer, program):
        return True
    return bool(anti_cheat(answer, ""))


def _verify_answer(answer: str, frozen_prefix: str, resource_limit: int) -> tuple[bool, float]:
    """Verify one attempt: ``(verified, Dafny seconds)``; a cheat never reaches Dafny.

    This runs in process-pool workers, so it stays a module-level function taking picklable
    arguments only. A broken Dafny tool raises ``ToolError`` in the worker and aborts the run.
    """
    if _is_cheat(answer, frozen_prefix):
        return False, 0.0
    outcome = verify(answer, resource_limit=resource_limit)
    return outcome.verified, outcome.seconds


def _verify_proof_only_answer(answer: str, program: str, resource_limit: int) -> tuple[bool, float]:
    """Verify one proof-only attempt: ``(verified, Dafny seconds)``; a cheat never reaches Dafny.

    This runs in process-pool workers, so it stays a module-level function taking picklable
    arguments only. A broken Dafny tool raises ``ToolError`` in the worker and aborts the run.
    """
    if _is_proof_only_cheat(answer, program):
        return False, 0.0
    outcome = verify(answer, resource_limit=resource_limit)
    return outcome.verified, outcome.seconds


def _sample_task(task: Task, client: ModelClient, model: str, attempts: int) -> list[_Attempt]:
    """Sample ``attempts`` independent answers for ``task``; a ``ToolError`` escapes to the caller."""
    if _is_proof_only(task):
        program = task.payload["program"]
        assert isinstance(program, str)
        messages = (Message("system", SYSTEM_MESSAGE), Message("user", build_proof_only_prompt(program)))
    else:
        messages = (Message("system", SYSTEM_MESSAGE), Message("user", build_prompt(task)))
    found: list[_Attempt] = []
    for _ in range(attempts):
        completion = client.complete(model, messages, step=_STEP, temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS)
        found.append(_Attempt(completion.text, completion.usage.cost_usd, completion.usage.seconds))
    return found


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    """The parsed JSON lines of ``path``, or an empty list when it does not exist yet (resume)."""
    if not path.is_file():
        return []
    try:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except ValueError as exc:
        raise ToolError(f"cannot resume from {label} {path}: {exc}") from exc
    for line in lines:
        if not isinstance(line, dict):
            raise ToolError(f"cannot resume from {label} {path}: expected JSON objects")
    return lines


def _done_lines(out_dir: Path) -> dict[str, dict[str, Any]]:
    """The completed task lines of a previous partial run, keyed by task ID."""
    done: dict[str, dict[str, Any]] = {}
    for line in _read_jsonl(out_dir / "results.jsonl", "results"):
        task_id = line.get("task_id", line.get("id"))
        if isinstance(task_id, str) and task_id and task_id not in done:
            done[task_id] = line
    return done


def _default_git_sha() -> str:
    """The repository HEAD commit; a missing or broken git is a ``ToolError``."""
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"could not determine the repository HEAD commit: {exc}") from exc
    sha = proc.stdout.strip()
    if proc.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        detail = (proc.stderr or proc.stdout).strip() or f"git rev-parse exited with code {proc.returncode}"
        raise ToolError(f"could not determine the repository HEAD commit: {detail}")
    return sha


def _default_started_utc() -> str:
    """The current UTC time as an ISO-8601 ``Z`` timestamp."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split_sha256() -> str:
    """SHA-256 of the raw committed ``vericoding_test.txt`` bytes."""
    try:
        raw = SPLIT_FILE.read_bytes()
    except OSError as exc:
        raise ToolError(f"could not read the committed split file {SPLIT_FILE}: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_tasks_file(path: Path) -> tuple[Task, ...]:
    """Proof-only tasks from a ``--tasks-file`` JSONL file.

    Every line is exactly a task object with nonempty string ``id``,
    ``source_task`` and ``program``; anything else raises a ``ToolError`` naming
    the file and the line, without echoing the line itself.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolError(f"cannot read tasks file {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"cannot read tasks file {path}: {exc}") from exc
    tasks: list[Task] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        label = f"{path} line {lineno}"
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ToolError(f"cannot parse tasks file {label}: {exc}") from exc
        if not isinstance(record, dict):
            raise ToolError(f"cannot parse tasks file {label}: expected a JSON object")
        task_id = record.get("id")
        source_task = record.get("source_task")
        program = record.get("program")
        if not isinstance(task_id, str) or not task_id:
            raise ToolError(f"cannot parse tasks file {label}: expected a nonempty string id")
        if not isinstance(source_task, str) or not source_task:
            raise ToolError(f"cannot parse tasks file {label}: expected a nonempty string source_task")
        if not isinstance(program, str) or not program:
            raise ToolError(f"cannot parse tasks file {label}: expected a nonempty string program")
        if task_id in seen:
            raise ToolError(f"cannot parse tasks file {label}: duplicate task id")
        seen.add(task_id)
        tasks.append(Task(id=task_id, source_task=source_task, payload={"program": program}))
    return tuple(tasks)


def _load_resume(
    out_dir: Path,
    split: str,
    split_sha256: str,
    endpoint: str,
    model: str,
    attempts: int,
    max_cost_usd: float,
    contributor: bool,
    started_utc: str | None,
    git_sha: str | None,
) -> tuple[dict[str, dict[str, Any]], float, str, str, str]:
    """The (done lines, spent cost, started_utc, git_sha, split_sha256) of a previous partial run.

    Resuming with different harness options would silently mix results from another
    configuration into the metrics, so a mismatch in ``split`` (the test/train name
    or the resolved tasks-file path), ``split_sha256`` (the committed split or the
    tasks file bytes), ``endpoint``, ``model``, ``attempts``, ``max_cost_usd`` or
    ``contributor`` raises a ``ToolError`` instead. The run
    metadata (``started_utc``, ``git_sha``, ``split_sha256``) stays stable across resumes.
    """
    done = _done_lines(out_dir)
    spent = sum(float(line.get("cost_usd", 0.0)) for line in done.values())
    run_file = out_dir / "run.json"
    if not run_file.is_file():
        return (
            done,
            spent,
            started_utc or _default_started_utc(),
            git_sha or _default_git_sha(),
            split_sha256,
        )
    try:
        previous = json.loads(run_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ToolError(f"cannot resume from {run_file}: {exc}") from exc
    if not isinstance(previous, dict):
        raise ToolError(f"cannot resume from {run_file}: expected a JSON object")
    if previous.get("harness") != HARNESS or previous.get("split") != split:
        raise ToolError(f"cannot resume from {run_file}: not a {HARNESS} {split} run")
    for key, value in (
        ("endpoint", endpoint),
        ("model", model),
        ("attempts", attempts),
        ("max_cost_usd", max_cost_usd),
        ("contributor", contributor),
    ):
        if previous.get(key) != value:
            raise ToolError(
                f"cannot resume from {run_file}: {key} mismatch "
                f"({previous.get(key)!r} != {value!r}); use a fresh --out directory"
            )
    if previous.get("split_sha256") != split_sha256:
        raise ToolError(f"cannot resume from {run_file}: the split content changed; use a fresh --out directory")
    kept_utc = previous.get("started_utc")
    kept_sha = previous.get("git_sha")
    if not isinstance(kept_utc, str) or not kept_utc:
        raise ToolError(f"cannot resume from {run_file}: previous run has no started_utc")
    if not isinstance(kept_sha, str) or not kept_sha:
        raise ToolError(f"cannot resume from {run_file}: previous run has no git_sha")
    return done, spent, kept_utc, kept_sha, split_sha256


def _partition_eligible(tasks: list[Task]) -> tuple[list[Task], list[str]]:
    """Split ``tasks`` into eligible tasks and sorted skipped IDs.

    A task is skipped only when ``build_prompt`` raises ``ValueError`` (for
    example Vericoding rows with an empty ``vc-spec`` and no prompt stub); any
    other error propagates. Proof-only tasks use ``build_proof_only_prompt``
    instead, so they are never probed with ``build_prompt``.
    """
    eligible: list[Task] = []
    skipped: list[str] = []
    for task in tasks:
        if _is_proof_only(task):
            eligible.append(task)
            continue
        try:
            build_prompt(task)
        except ValueError:
            skipped.append(task.id)
        else:
            eligible.append(task)
    return eligible, sorted(skipped)


def _write_skipped_report(out_dir: Path, *, total_tasks: int, eligible_tasks: int, skipped_ids: list[str]) -> None:
    """Write ``skipped.json`` with the deterministic skip report of this run."""
    (out_dir / "skipped.json").write_text(
        json.dumps(
            {
                "total_tasks": total_tasks,
                "eligible_tasks": eligible_tasks,
                "skipped_count": len(skipped_ids),
                "skipped_ids": sorted(skipped_ids),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _metrics(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """The counts, proof rate and totals of completed result lines."""
    verified = sum(1 for line in lines if line.get("verified") is True)
    total = len(lines)
    return {
        "total": total,
        "verified": verified,
        "proof_rate": (verified / total) if total else 0.0,
        "total_cost_usd": sum(float(line.get("cost_usd", 0.0)) for line in lines),
        "total_seconds": sum(float(line.get("seconds", 0.0)) for line in lines),
    }


def _result_line(
    task_id: str,
    verified: bool,
    attempts: int,
    verified_attempts: int,
    cost_usd: float,
    seconds: float,
    answer: str,
) -> dict[str, Any]:
    """One ``results.jsonl`` line; ``id`` and ``task_id`` carry the same task ID."""
    return {
        "id": task_id,
        "task_id": task_id,
        "verified": verified,
        "attempts": attempts,
        "verified_attempts": verified_attempts,
        "cost_usd": cost_usd,
        "seconds": seconds,
        "answer": answer,
    }


def _write_run_json(
    out_dir: Path,
    *,
    endpoint: str,
    model: str,
    attempts: int,
    split: str,
    split_sha256: str,
    contributor: bool,
    git_sha: str,
    started_utc: str,
    max_cost_usd: float,
    stopped_early: bool,
    status: str,
    task_ids: list[str],
    metrics: dict[str, Any] | None,
    total_tasks: int | None = None,
    skipped_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Write ``run.json`` and return its content.

    Besides the protocol fields this records ``date``/``commit``/``dataset`` mirrors, the task
    IDs, the stop flag and the metrics, so the run publishes with ``pcc.evaluation.report``.
    """
    summary: dict[str, Any] = {
        "harness": HARNESS,
        "endpoint": endpoint,
        "model": model,
        "attempts": attempts,
        "split": split,
        "split_sha256": split_sha256,
        "contributor": contributor,
        "git_sha": git_sha,
        "started_utc": started_utc,
        "max_cost_usd": max_cost_usd,
        "date": started_utc[:10],
        "commit": git_sha,
        "dataset": f"{HARNESS}/{split}",
        "stopped_early": stopped_early,
        "status": status,
        "tasks": task_ids,
        "metrics": metrics,
        "total_tasks": total_tasks if total_tasks is not None else len(task_ids),
        "eligible_tasks": len(task_ids),
        "skipped_count": len(skipped_ids) if skipped_ids is not None else 0,
        "skipped_ids": sorted(skipped_ids) if skipped_ids is not None else [],
    }
    (out_dir / "run.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def _write_summary(
    out_dir: Path,
    *,
    endpoint: str,
    model: str,
    attempts: int,
    split: str,
    lines: list[dict[str, Any]],
    stopped: bool,
    total_tasks: int | None = None,
    skipped_ids: list[str] | None = None,
) -> None:
    """Write ``summary.md``: the proof rate, the totals and the cost and time per task."""
    metrics = _metrics(lines)
    skipped = sorted(skipped_ids) if skipped_ids is not None else []
    total = total_tasks if total_tasks is not None else metrics["total"]
    text = [
        f"# Vericoding run: {out_dir.name}",
        "",
        f"- Model: {model} (endpoint {endpoint})",
        f"- Split: {split}",
        f"- Attempts per task: {attempts}",
        f"- Eligible tasks: {metrics['total']} of {total} (skipped {len(skipped)})",
        f"- Verified: {metrics['verified']} of {metrics['total']} (proof rate {metrics['proof_rate']:.3f})",
        f"- Total cost: {metrics['total_cost_usd']:.2f} USD",
        f"- Total time: {metrics['total_seconds']:.1f} s",
    ]
    if skipped:
        text.append(f"- Skipped tasks (unbuildable prompt): {', '.join(skipped)}")
    if stopped:
        text.append(f"- {CAP_LINE}")
    text += ["", "## Tasks", ""]
    for line in lines:
        state = "verified" if line.get("verified") is True else "unverified"
        text.append(
            f"- {line.get('task_id')}: {state} "
            f"(attempts={line.get('attempts')}, verified_attempts={line.get('verified_attempts')}, "
            f"cost_usd={float(line.get('cost_usd', 0.0)):.2f}, seconds={float(line.get('seconds', 0.0)):.1f})"
        )
    (out_dir / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run_harness(
    *,
    split: str | None = None,
    out_dir: Path,
    model: str,
    attempts: int,
    max_cost_usd: float,
    endpoint: str = "meta",
    concurrency: int = 16,
    workers: int | None = None,
    client: ModelClient | None = None,
    config: Config | None = None,
    tasks: tuple[Task, ...] | None = None,
    tasks_file: Path | None = None,
    verify_fn: Callable[[str, str], tuple[bool, float]] | None = None,
    git_sha: str | None = None,
    started_utc: str | None = None,
) -> dict[str, Any]:
    """Run the Vericoding harness and return the ``run.json`` summary.

    ``tasks`` overrides the dataset split (a test hook that avoids downloads); ``client``
    overrides the model client (tests pass a scripted client); ``verify_fn`` replaces the
    process-pool Dafny verification with ``(verified, seconds)`` per answer (a test hook —
    production always verifies with ``verifier.verify``); ``workers`` sizes the process pool
    (the default is ``os.cpu_count()``); ``git_sha`` and ``started_utc`` override the
    detected run metadata. ``verify_fn`` still sees no cheat: answers that fail the frozen-prefix
    contract never reach it.

    With ``tasks_file`` (mutually exclusive with ``split``) the harness runs proof-only
    tasks from a JSONL file instead of a split: every line holds ``id``,
    ``source_task`` and ``program``. The user prompt is
    ``proof_only.build_proof_only_prompt(program)``, and an answer counts as
    verified only when it keeps the program's executable code and header, passes
    the cheat scan and verifies with Dafny. ``run.json`` records the resolved
    tasks-file path as ``split`` and the SHA-256 of its raw bytes as
    ``split_sha256``; resuming requires both to match.

    Tasks whose prompt cannot be built (``build_prompt`` raises ``ValueError``,
    e.g. Vericoding rows with an empty ``vc-spec`` and no prompt stub) are
    skipped deterministically before any sampling, on every invocation including
    resumes: they are never sampled, verified, costed or recorded in
    ``results.jsonl``. The skip is reported in ``<out_dir>/skipped.json``
    (``total_tasks``, ``eligible_tasks``, ``skipped_count``, sorted
    ``skipped_ids``) and in ``run.json`` and ``summary.md``; ``run.json`` tasks
    and metrics count only eligible tasks. Only ``ValueError`` from
    ``build_prompt`` is skipped: any other error still propagates.
    """
    if tasks_file is not None and split is not None:
        raise ValueError("--tasks-file is mutually exclusive with --split")
    if tasks_file is None:
        if split is None:
            raise ValueError("either --tasks-file or --split is required")
        if split == "test":
            which: Literal["test", "train"] = "test"
        elif split == "train":
            which = "train"
        else:
            raise ValueError(f"unknown split {split!r}; expected 'test' or 'train'")
        split_label = split
        file_sha = _split_sha256()
    else:
        resolved_file = tasks_file.resolve()
        try:
            file_bytes = resolved_file.read_bytes()
        except OSError as exc:
            raise ToolError(f"cannot read tasks file {resolved_file}: {exc}") from exc
        split_label = str(resolved_file)
        file_sha = hashlib.sha256(file_bytes).hexdigest()
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    effective = config or Config()
    stage = ProverStage(name=endpoint, model=model, attempts=attempts, endpoint=endpoint)
    stage_config = dataclasses.replace(effective, proof=dataclasses.replace(effective.proof, prover_stages=(stage,)))
    if client is None:
        router = models.build_client(stage_config)
        if not router.available(model):
            raise ModelUnavailable(f"model {model!r} is not available through endpoint {endpoint!r}")
        active: ModelClient = router
    else:
        active = client
    contributor = model == _CONTRIBUTOR_MODEL
    if tasks_file is None:
        resolved = tasks if tasks is not None else datasets.split(datasets.VERICODING, which)
    elif tasks is not None:
        resolved = tasks
    else:
        resolved = _load_tasks_file(resolved_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_all = sorted(resolved, key=lambda task: task.id)
    ordered, skipped_ids = _partition_eligible(ordered_all)
    task_ids = [task.id for task in ordered]
    _write_skipped_report(out_dir, total_tasks=len(ordered_all), eligible_tasks=len(ordered), skipped_ids=skipped_ids)
    done, spent, kept_utc, kept_sha, split_sha = _load_resume(
        out_dir, split_label, file_sha, endpoint, model, attempts, max_cost_usd, contributor, started_utc, git_sha
    )
    _write_run_json(
        out_dir,
        endpoint=endpoint,
        model=model,
        attempts=attempts,
        split=split_label,
        split_sha256=split_sha,
        contributor=contributor,
        git_sha=kept_sha,
        started_utc=kept_utc,
        max_cost_usd=max_cost_usd,
        stopped_early=False,
        status="running",
        task_ids=task_ids,
        metrics=None,
        total_tasks=len(ordered_all),
        skipped_ids=skipped_ids,
    )
    pending = [task for task in ordered if task.id not in done]
    stopped = False
    try:
        sampled = _sample_pending(pending, active, model, attempts, concurrency, max_cost_usd, spent)
        if len(sampled) < len(pending):
            stopped = True
        results_path = out_dir / "results.jsonl"
        with results_path.open("a", encoding="utf-8") as handle:
            if verify_fn is not None:
                for task in ordered:
                    if task.id in done or task.id not in sampled:
                        continue
                    checks = _check_inline(task, sampled[task.id], verify_fn)
                    line = _score_task(task, sampled[task.id], checks)
                    handle.write(json.dumps(line, sort_keys=True) + "\n")
                    done[task.id] = line
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    future_of: dict[str, list[Future[tuple[bool, float]]]] = {}
                    for task in ordered:
                        if task.id in done or task.id not in sampled:
                            continue
                        context = task_spec_of(task)
                        if _is_proof_only(task):
                            future_of[task.id] = [
                                pool.submit(
                                    _verify_proof_only_answer,
                                    attempt.text,
                                    context,
                                    effective.dafny.resource_limit,
                                )
                                for attempt in sampled[task.id]
                            ]
                        else:
                            future_of[task.id] = [
                                pool.submit(_verify_answer, attempt.text, context, effective.dafny.resource_limit)
                                for attempt in sampled[task.id]
                            ]
                    for task in ordered:
                        if task.id not in future_of:
                            continue
                        checks = [future.result() for future in future_of[task.id]]
                        line = _score_task(task, sampled[task.id], checks)
                        handle.write(json.dumps(line, sort_keys=True) + "\n")
                        done[task.id] = line
    except ToolError:
        lines = [done[task.id] for task in ordered if task.id in done]
        _write_run_json(
            out_dir,
            endpoint=endpoint,
            model=model,
            attempts=attempts,
            split=split_label,
            split_sha256=split_sha,
            contributor=contributor,
            git_sha=kept_sha,
            started_utc=kept_utc,
            max_cost_usd=max_cost_usd,
            stopped_early=stopped,
            status="error",
            task_ids=task_ids,
            metrics=_metrics(lines),
            total_tasks=len(ordered_all),
            skipped_ids=skipped_ids,
        )
        _write_summary(
            out_dir,
            endpoint=endpoint,
            model=model,
            attempts=attempts,
            split=split_label,
            lines=lines,
            stopped=stopped,
            total_tasks=len(ordered_all),
            skipped_ids=skipped_ids,
        )
        raise
    lines = [done[task.id] for task in ordered if task.id in done]
    summary = _write_run_json(
        out_dir,
        endpoint=endpoint,
        model=model,
        attempts=attempts,
        split=split_label,
        split_sha256=split_sha,
        contributor=contributor,
        git_sha=kept_sha,
        started_utc=kept_utc,
        max_cost_usd=max_cost_usd,
        stopped_early=stopped,
        status="done",
        task_ids=task_ids,
        metrics=_metrics(lines),
        total_tasks=len(ordered_all),
        skipped_ids=skipped_ids,
    )
    _write_summary(
        out_dir,
        endpoint=endpoint,
        model=model,
        attempts=attempts,
        split=split_label,
        lines=lines,
        stopped=stopped,
        total_tasks=len(ordered_all),
        skipped_ids=skipped_ids,
    )
    return summary


def _sample_pending(
    pending: list[Task],
    client: ModelClient,
    model: str,
    attempts: int,
    concurrency: int,
    max_cost_usd: float,
    spent: float,
) -> dict[str, list[_Attempt]]:
    """Sample every pending task in a thread pool, in task-ID order of submission.

    No new task is submitted once the completed cost reaches ``max_cost_usd``; tasks already
    submitted finish and their actual cost counts. Returns the attempts of sampled tasks only.
    """
    sampled: dict[str, list[_Attempt]] = {}
    queue = list(pending)
    running = spent
    with ThreadPoolExecutor(max_workers=concurrency) as sampler:
        in_flight: dict[Future[list[_Attempt]], Task] = {}
        while queue and len(in_flight) < concurrency and running < max_cost_usd:
            task = queue.pop(0)
            in_flight[sampler.submit(_sample_task, task, client, model, attempts)] = task
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                task = in_flight.pop(future)
                attempts_done = future.result()
                running += sum(attempt.cost_usd for attempt in attempts_done)
                sampled[task.id] = attempts_done
            while queue and len(in_flight) < concurrency and running < max_cost_usd:
                task = queue.pop(0)
                in_flight[sampler.submit(_sample_task, task, client, model, attempts)] = task
    return sampled


def _check_inline(
    task: Task,
    attempts_done: list[_Attempt],
    verify_fn: Callable[[str, str], tuple[bool, float]],
) -> list[tuple[bool, float]]:
    """Check sampled attempts without Dafny: cheats fail, the rest go to ``verify_fn`` (tests)."""
    if _is_proof_only(task):
        program = task.payload["program"]
        assert isinstance(program, str)
        checks: list[tuple[bool, float]] = []
        for attempt in attempts_done:
            if _is_proof_only_cheat(attempt.text, program):
                checks.append((False, 0.0))
            else:
                checks.append(verify_fn(attempt.text, program))
        return checks
    frozen = task_spec_of(task)
    checks = []
    for attempt in attempts_done:
        if _is_cheat(attempt.text, frozen):
            checks.append((False, 0.0))
        else:
            checks.append(verify_fn(attempt.text, frozen))
    return checks


def _score_task(task: Task, attempts_done: list[_Attempt], checks: list[tuple[bool, float]]) -> dict[str, Any]:
    """One result line: verified when at least one attempt verifies, else the last attempt."""
    verified_attempts = sum(1 for passed, _ in checks if passed)
    first_good = next(
        (attempt.text for attempt, (passed, _) in zip(attempts_done, checks, strict=True) if passed),
        None,
    )
    answer = first_good if first_good is not None else attempts_done[-1].text
    cost_usd = sum(attempt.cost_usd for attempt in attempts_done)
    seconds = sum(attempt.seconds for attempt in attempts_done) + sum(check_seconds for _, check_seconds in checks)
    return _result_line(
        task.id, verified_attempts >= 1, len(attempts_done), verified_attempts, cost_usd, seconds, answer
    )


__all__ = ["CAP_LINE", "HARNESS", "SPLIT_FILE", "run_harness", "task_spec_of"]
