"""HumanEvalFix harness: the whole pipeline on each task, in its fixed and buggy versions.

Each task becomes a temporary repository holding the function as ``solution.py`` and the task's tests as
``tests/test_task.py`` with one ``def test_<entry_point>():``. The pipeline runs on both versions. In
``--spec-only`` mode only steps 1 to 3 run (collect, derive, write, check with the MUTANTS rewrite rule) and each
version gets one outcome line. Every pipeline result is also appended to ``scan.jsonl`` so the results viewer can
browse the run; with ``--split train`` the proof prompt of each proven result is collected into ``proofs.jsonl``.

The output files (``run.json``, ``scan.jsonl``/``spec_only.jsonl``, ``proofs.jsonl``) are append-friendly: rerunning
into the same directory skips tasks that already have a line (resume). ``run.json`` holds the harness name, the
options and the metrics; tasks whose fixed version fails its own tests are excluded and listed there.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pcc import context, intent, models, pipeline, sandbox
from pcc import spec as spec_step
from pcc.config import Config
from pcc.core import (
    CheckKind,
    FunctionRef,
    NotCheckable,
    Result,
    Spec,
    State,
    ToolError,
    TranslationPath,
    source_hash,
    to_data,
)
from pcc.evaluation import datasets
from pcc.models import Message, ModelClient, Router
from pcc.models.client import Completion

HARNESS = "humanevalfix"

FIXED: Literal["fixed"] = "fixed"
BUGGY: Literal["buggy"] = "buggy"

Version = Literal["fixed", "buggy"]

_PROJECT_OF_VERSION = {FIXED: "humanevalfix-fixed", BUGGY: "humanevalfix-buggy"}
_SOLUTION_OF_VERSION = {FIXED: "canonical_solution", BUGGY: "buggy_solution"}

_SPARK_STANDARD = "muse-spark-1.3"
_SPARK_CONTRIBUTOR = "muse-spark-1.3-contributor"

_TASK_TESTS_MARKER = "PCC_TASK_TESTS_PASSED"

_OUTCOMES = ("pass", "false_alarm", "caught", "missed", "not_checkable")


class TaskSkipped(Exception):
    """A task that cannot be evaluated; it is excluded and listed instead of failing the run."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _RecordedCall:
    """One model call seen by the recorder, with its answer and cost."""

    model: str
    messages: tuple[Message, ...]
    step: str
    answer: str
    usage_cost_usd: float
    usage_seconds: float


class _Recorder:
    """A ``ModelClient`` wrapper recording every call; feeds cost accounting and ``proofs.jsonl``."""

    def __init__(self, inner: ModelClient) -> None:
        self._inner = inner
        self.calls: list[_RecordedCall] = []

    def complete(
        self,
        model: str,
        messages: tuple[Message, ...],
        *,
        step: str,
        schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        completion = self._inner.complete(
            model, messages, step=step, schema=schema, temperature=temperature, max_tokens=max_tokens
        )
        self.calls.append(
            _RecordedCall(model, messages, step, completion.text, completion.usage.cost_usd, completion.usage.seconds)
        )
        return completion

    def cost_since(self, index: int) -> tuple[float, float]:
        """The (cost, seconds) of calls recorded from ``index`` on."""
        calls = self.calls[index:]
        return sum(call.usage_cost_usd for call in calls), sum(call.usage_seconds for call in calls)


def _contributor_model(model: str) -> str:
    """``model`` with the contributor rule applied: standard Spark becomes contributor Spark."""
    return _SPARK_CONTRIBUTOR if model == _SPARK_STANDARD else model


def apply_options(config: Config, *, stages: str | None, contributor: bool, spec_model: str | None) -> Config:
    """The config with harness options applied: ``--spec-model``, ``--contributor``, ``--stages``.

    ``spec_model`` sets ``ModelsConfig.spec`` first so the contributor rule applies to it too. ``contributor``
    replaces every standard Spark model (in every ``ModelsConfig`` field and in the prover stages) by the
    contributor one. ``stages`` keeps only the prover stage with that name; an unknown name raises
    ``ValueError``.
    """
    models_config = config.models
    if spec_model is not None:
        models_config = dataclasses.replace(models_config, spec=spec_model)
    if contributor:
        models_config = dataclasses.replace(
            models_config,
            intent=_contributor_model(models_config.intent),
            spec=_contributor_model(models_config.spec),
            mutants=_contributor_model(models_config.mutants),
            translation=_contributor_model(models_config.translation),
            edge_cases=_contributor_model(models_config.edge_cases),
            counterexample=_contributor_model(models_config.counterexample),
            second_opinion=_contributor_model(models_config.second_opinion),
            context_search=_contributor_model(models_config.context_search),
        )
    prover_stages = config.proof.prover_stages
    if contributor:
        prover_stages = tuple(
            dataclasses.replace(stage, model=_contributor_model(stage.model)) for stage in prover_stages
        )
    if stages is not None:
        kept = tuple(stage for stage in prover_stages if stage.name == stages)
        if not kept:
            names = ", ".join(stage.name for stage in prover_stages) or "none"
            raise ValueError(f"unknown stage {stages!r}; expected one of: {names}")
        prover_stages = kept
    proof_config = dataclasses.replace(config.proof, prover_stages=prover_stages)
    return dataclasses.replace(config, models=models_config, proof=proof_config)


def entry_point_of(task: datasets.Task) -> str:
    """The function name under test; a missing one excludes the task instead of failing the run."""
    entry_point = task.payload.get("entry_point")
    if not isinstance(entry_point, str) or not entry_point:
        raise TaskSkipped(f"task {task.id} has no entry point")
    return entry_point


def solution_source(task: datasets.Task, version: Version) -> str:
    """The full function source of one version: declaration plus the fixed or buggy body."""
    declaration = task.payload.get("declaration")
    body = task.payload.get(_SOLUTION_OF_VERSION[version])
    if not isinstance(declaration, str) or not isinstance(body, str):
        raise TaskSkipped(f"task {task.id} has no {version} solution")
    return declaration.rstrip("\n") + "\n" + body.rstrip("\n") + "\n"


def check_body(task: datasets.Task) -> tuple[str, list[str]]:
    """The (parameter, assert lines) of the task's check function; unparseable tests exclude the task."""
    raw = task.payload.get("test")
    if not isinstance(raw, str):
        raise TaskSkipped(f"task {task.id} has no test source")
    try:
        tree = ast.parse(raw)
    except SyntaxError as exc:
        raise TaskSkipped(f"task {task.id} has an unparseable test: {exc}") from exc
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = [*node.args.posonlyargs, *node.args.args]
        if (
            len(positional) != 1
            or node.args.vararg is not None
            or node.args.kwonlyargs
            or node.args.kwarg is not None
            or node.args.defaults
            or node.args.kw_defaults
        ):
            raise TaskSkipped(f"task {task.id} has a check function with an unexpected signature")
        statements = [ast.unparse(statement) for statement in node.body]
        if not statements:
            raise TaskSkipped(f"task {task.id} has an empty test")
        return positional[0].arg, statements
    raise TaskSkipped(f"task {task.id} has no check function in its test")


def test_module_source(task: datasets.Task) -> str:
    """The content of ``tests/test_task.py``: one ``def test_<entry_point>():`` calling the function by name."""
    entry_point = entry_point_of(task)
    param, statements = check_body(task)
    lines: list[str] = []
    extra_import = task.payload.get("import")
    if isinstance(extra_import, str) and extra_import.strip():
        lines.append(extra_import.rstrip("\n"))
    test_setup = task.payload.get("test_setup")
    if isinstance(test_setup, str) and test_setup.strip():
        lines.append(test_setup.rstrip("\n"))
    if param == entry_point:
        lines.append(f"from solution import {entry_point}")
    else:
        lines.append(f"from solution import {entry_point} as {param}")
    lines.append("")
    lines.append(f"def test_{entry_point}():")
    for statement in statements:
        for line in statement.splitlines():
            lines.append(f"    {line}" if line.strip() else "")
    lines.append("")
    return "\n".join(lines)


def write_task_repo(task: datasets.Task, version: Version, root: Path) -> FunctionRef:
    """Create the temporary repository for one version and return its function reference.

    The layout is ``solution.py`` with the function plus ``tests/test_task.py`` with the task's tests.
    A task that cannot be converted raises ``TaskSkipped``.
    """
    entry_point = entry_point_of(task)
    source = solution_source(task, version)
    root.mkdir(parents=True, exist_ok=True)
    (root / "solution.py").write_text(source, encoding="utf-8")
    tests_dir = root / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test_task.py").write_text(test_module_source(task), encoding="utf-8")
    return FunctionRef("solution.py", entry_point, 1, source, source_hash(source))


def task_tests_pass(repo_root: Path, entry_point: str) -> bool:
    """Whether the task's own tests pass against the repository: the fixed version must pass these."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"sys.path.insert(0, {str(repo_root / 'tests')!r})\n"
        f"from test_task import test_{entry_point} as _task_test\n"
        "_task_test()\n"
        f"print({_TASK_TESTS_MARKER!r})\n"
    )
    outcome = sandbox.run_python(script, cwd=repo_root)
    return outcome.ok and _TASK_TESTS_MARKER in outcome.stdout


def _mutants_detail(checked: Spec) -> str | None:
    """The detail of the failed MUTANTS check, or None when the spec passed the mutant check."""
    for record in checked.checks:
        if record.kind is CheckKind.MUTANTS and not record.passed:
            return record.detail
    return None


def classify_spec(checked: Spec, version: Version) -> tuple[str, str]:
    """The spec-only (outcome, reason) from the frozen spec's hypothesis check on the original code."""
    hyp = next((record for record in checked.checks if record.kind is CheckKind.HYPOTHESIS_ORIGINAL), None)
    if hyp is None:
        return "not_checkable", "no hypothesis check recorded"
    if hyp.passed:
        return ("pass", "") if version == FIXED else ("missed", "")
    if hyp.detail in ("precondition never holds", "timeout"):
        return "not_checkable", hyp.detail
    if version == FIXED:
        return "false_alarm", hyp.detail
    return "caught", hyp.detail


def run_spec_only(
    task: datasets.Task, version: Version, repo_root: Path, *, recorder: _Recorder, config: Config
) -> dict[str, Any]:
    """Steps 1 to 3 on one version: collect, derive, write and check, rewriting at most ``max_rewrites`` times.

    The rewrite hint is the failed MUTANTS detail (inputs with the original's and the mutant's result, never
    mutant code). Translation, proof and second opinion never run here.
    """
    function = write_task_repo(task, version, repo_root)
    calls_before = len(recorder.calls)
    try:
        collected = context.collect(function, repo_root)
        derived = intent.derive(collected, client=recorder, model=config.models.intent)
        checked = spec_step.write(derived, client=recorder, model=config.models.spec)
        checked = spec_step.check(
            checked, function, repo_root, client=recorder, model=config.models.mutants, config=config.spec
        )
        rewrites = 0
        while (hint := _mutants_detail(checked)) is not None and rewrites < config.spec.max_rewrites:
            checked = spec_step.write(derived, client=recorder, model=config.models.spec, hint=hint)
            checked = spec_step.check(
                checked, function, repo_root, client=recorder, model=config.models.mutants, config=config.spec
            )
            rewrites += 1
    except NotCheckable as exc:
        outcome, reason = "not_checkable", str(exc) or "not checkable"
    else:
        outcome, reason = classify_spec(checked, version)
    cost_usd, seconds = recorder.cost_since(calls_before)
    return {
        "task_id": task.id,
        "version": version,
        "outcome": outcome,
        "reason": reason,
        "cost_usd": cost_usd,
        "seconds": seconds,
    }


def _result_cost(result: Result) -> tuple[float, float]:
    """The (cost, seconds) of a pipeline result from its usage records."""
    return sum(usage.cost_usd for usage in result.usage), sum(usage.seconds for usage in result.usage)


def scan_line(task_id: str, version: Version, result: Result, cost_usd: float, seconds: float) -> dict[str, Any]:
    """One ``scan.jsonl`` line for a pipeline result, with the project naming the evaluated version."""
    return {
        "project": _PROJECT_OF_VERSION[version],
        "task_id": task_id,
        "version": version,
        "state": result.state.value,
        "reason": result.reason,
        "name": result.function.qualname,
        "path": result.function.path,
        "cost_usd": cost_usd,
        "seconds": seconds,
        "result": to_data(result),
    }


def proof_line(task_id: str, call: _RecordedCall) -> dict[str, Any]:
    """One ``proofs.jsonl`` line: the proof prompt and answer that produced a proven train result."""
    return {
        "task_id": task_id,
        "messages": [{"role": message.role, "content": message.content} for message in call.messages],
        "answer": call.answer,
    }


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    """The parsed JSON lines of ``path``, or an empty list when it does not exist yet (resume)."""
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except ValueError as exc:
        raise ToolError(f"cannot resume from {label} {path}: {exc}") from exc


def _append_jsonl(path: Path, line: dict[str, Any] | Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def _load_resume(
    out_dir: Path,
    split: str,
    spec_only: bool,
    stages: str | None,
    spec_model: str | None,
    max_cost_usd: float,
    contributor: bool,
) -> tuple[set[tuple[str, str]], list[dict[str, str]], float]:
    """The (done task/version pairs, excluded tasks, spent cost) of a previous partial run, if any.

    Resuming with different harness options would silently mix results from another
    configuration into the metrics, so a mismatch in ``stages``, ``spec_model``,
    ``max_cost_usd`` or ``contributor`` raises a ``ToolError`` instead.
    """
    lines_file = out_dir / ("spec_only.jsonl" if spec_only else "scan.jsonl")
    done: set[tuple[str, str]] = set()
    spent = 0.0
    for line in _read_jsonl(lines_file, "results"):
        task_id = line.get("task_id")
        version = line.get("version")
        if isinstance(task_id, str) and isinstance(version, str):
            done.add((task_id, version))
        cost = line.get("cost_usd", 0.0)
        if isinstance(cost, (int, float)):
            spent += float(cost)
    excluded: list[dict[str, str]] = []
    run_file = out_dir / "run.json"
    if run_file.is_file():
        try:
            previous = json.loads(run_file.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ToolError(f"cannot resume from {run_file}: {exc}") from exc
        if not isinstance(previous, dict):
            raise ToolError(f"cannot resume from {run_file}: expected a JSON object")
        if previous.get("harness") != HARNESS or previous.get("split") != split:
            raise ToolError(f"cannot resume from {run_file}: not a {HARNESS} {split} run")
        if previous.get("spec_only") != spec_only:
            raise ToolError(f"cannot resume from {run_file}: spec_only mismatch")
        if previous.get("stages") != stages:
            raise ToolError(
                f"cannot resume from {run_file}: stages mismatch "
                f"({previous.get('stages')!r} != {stages!r}); use a fresh --out directory"
            )
        if previous.get("spec_model") != spec_model:
            raise ToolError(
                f"cannot resume from {run_file}: spec_model mismatch "
                f"({previous.get('spec_model')!r} != {spec_model!r}); use a fresh --out directory"
            )
        if previous.get("max_cost_usd") != max_cost_usd:
            raise ToolError(
                f"cannot resume from {run_file}: max_cost_usd mismatch "
                f"({previous.get('max_cost_usd')!r} != {max_cost_usd!r}); use a fresh --out directory"
            )
        if previous.get("contributor", False) != contributor:
            raise ToolError(
                f"cannot resume from {run_file}: contributor mismatch "
                f"({previous.get('contributor', False)!r} != {contributor!r}); use a fresh --out directory"
            )
        raw_excluded = previous.get("excluded", [])
        if isinstance(raw_excluded, list):
            excluded = [entry for entry in raw_excluded if isinstance(entry, dict)]
    return done, excluded, spent


def _translation_path(line: dict[str, Any]) -> str | None:
    result = line.get("result")
    if not isinstance(result, dict):
        return None
    translation = result.get("translation")
    if not isinstance(translation, dict):
        return None
    path = translation.get("function_path")
    return path if isinstance(path, str) else None


def _metrics_full(lines: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = [line for line in lines if line.get("version") == FIXED]
    buggy = [line for line in lines if line.get("version") == BUGGY]
    proven_fixed = sum(1 for line in fixed if line.get("state") == State.PROVEN.value)
    with_translation = [line for line in lines if _translation_path(line) is not None]
    transpiled = sum(1 for line in with_translation if _translation_path(line) == TranslationPath.TRANSPILER.value)
    total_cost = sum(float(line.get("cost_usd", 0.0)) for line in lines)
    total_seconds = sum(float(line.get("seconds", 0.0)) for line in lines)
    count = len(lines)
    return {
        "proof_rate_fixed": (proven_fixed / len(fixed)) if fixed else None,
        "bugs_found": sum(1 for line in buggy if line.get("state") == State.COUNTEREXAMPLE.value),
        "false_alarms": sum(1 for line in fixed if line.get("state") == State.COUNTEREXAMPLE.value),
        "transpiler_share": (transpiled / len(with_translation)) if with_translation else None,
        "total_cost_usd": total_cost,
        "total_seconds": total_seconds,
        "mean_cost_usd_per_function": (total_cost / count) if count else 0.0,
        "mean_seconds_per_function": (total_seconds / count) if count else 0.0,
        "evaluated": count,
    }


def _metrics_spec_only(lines: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = {outcome: 0 for outcome in _OUTCOMES}
    for line in lines:
        outcome = line.get("outcome")
        if outcome in outcomes:
            outcomes[outcome] += 1
    total_cost = sum(float(line.get("cost_usd", 0.0)) for line in lines)
    total_seconds = sum(float(line.get("seconds", 0.0)) for line in lines)
    return {"outcomes": outcomes, "total_cost_usd": total_cost, "total_seconds": total_seconds, "evaluated": len(lines)}


def _write_run_json(
    out_dir: Path,
    *,
    split: str,
    max_cost_usd: float,
    stages: str | None,
    spec_model: str | None,
    contributor: bool,
    spec_only: bool,
    task_ids: list[str],
    excluded: list[dict[str, str]],
    stopped_early: bool,
    status: str,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "harness": HARNESS,
        "split": split,
        "max_cost_usd": max_cost_usd,
        "stages": stages,
        "spec_model": spec_model,
        "contributor": contributor,
        "spec_only": spec_only,
        "tasks": task_ids,
        "excluded": excluded,
        "stopped_early": stopped_early,
        "status": status,
        "metrics": metrics,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def run_harness(
    *,
    split: str,
    out_dir: Path,
    max_cost_usd: float,
    stages: str | None = None,
    contributor: bool = False,
    spec_model: str | None = None,
    spec_only: bool = False,
    client: ModelClient | None = None,
    config: Config | None = None,
    tasks: tuple[datasets.Task, ...] | None = None,
) -> dict[str, Any]:
    """Run the HumanEvalFix harness and return the ``run.json`` summary.

    ``tasks`` overrides the dataset split (a test hook that avoids downloads); ``client`` overrides the model
    client (tests pass a ``FakeClient``). With ``spec_only`` each version gets one outcome line in
    ``spec_only.jsonl``; otherwise each pipeline result gets one line in ``scan.jsonl``.
    """
    if split == "test":
        which: Literal["test", "train"] = "test"
    elif split == "train":
        which = "train"
    else:
        raise ValueError(f"unknown split {split!r}; expected 'test' or 'train'")
    effective = apply_options(config or Config(), stages=stages, contributor=contributor, spec_model=spec_model)
    if client is None:
        router = models.build_client(effective)
        inner: ModelClient = router
    elif isinstance(client, Router):
        router = client
        inner = client
    else:
        allow_meta = {"META_API_KEY": "harness", "PCC_META_ALLOWED": "1"}
        router = models.build_client(effective, env=allow_meta, clients={"meta": client, "glimmer": client})
        inner = client
    filtered = models.available_stages(effective, router)
    recorder = _Recorder(inner)
    resolved = tasks if tasks is not None else datasets.split(HARNESS, which)
    out_dir.mkdir(parents=True, exist_ok=True)
    done, excluded, spent = _load_resume(out_dir, split, spec_only, stages, spec_model, max_cost_usd, contributor)
    excluded_ids = {entry.get("task_id") for entry in excluded}
    task_ids = [task.id for task in resolved]
    _write_run_json(
        out_dir,
        split=split,
        max_cost_usd=max_cost_usd,
        stages=stages,
        spec_model=spec_model,
        contributor=contributor,
        spec_only=spec_only,
        task_ids=task_ids,
        excluded=excluded,
        stopped_early=False,
        status="running",
        metrics=None,
    )
    lines_file = out_dir / ("spec_only.jsonl" if spec_only else "scan.jsonl")
    proofs_file = out_dir / "proofs.jsonl"
    stopped_early = False
    for task in resolved:
        if task.id in excluded_ids:
            continue
        try:
            entry_point = entry_point_of(task)
        except TaskSkipped as exc:
            excluded.append({"task_id": task.id, "reason": exc.reason})
            excluded_ids.add(task.id)
            continue
        with tempfile.TemporaryDirectory(prefix="humanevalfix-") as tmp:
            tmp_root = Path(tmp)
            fixed_dir = tmp_root / FIXED
            buggy_dir = tmp_root / BUGGY
            try:
                fixed_ref = write_task_repo(task, FIXED, fixed_dir)
                buggy_ref = write_task_repo(task, BUGGY, buggy_dir)
            except TaskSkipped as exc:
                excluded.append({"task_id": task.id, "reason": exc.reason})
                excluded_ids.add(task.id)
                continue
            if not task_tests_pass(fixed_dir, entry_point):
                excluded.append({"task_id": task.id, "reason": "fixed version fails its own tests"})
                excluded_ids.add(task.id)
                continue
            versions: tuple[tuple[Version, FunctionRef, Path], ...] = (
                (FIXED, fixed_ref, fixed_dir),
                (BUGGY, buggy_ref, buggy_dir),
            )
            for version, ref, repo_root in versions:
                if (task.id, version) in done:
                    continue
                if spent >= max_cost_usd:
                    stopped_early = True
                    break
                if spec_only:
                    line = run_spec_only(task, version, repo_root, recorder=recorder, config=filtered)
                    spent += float(line["cost_usd"])
                    done.add((task.id, version))
                    _append_jsonl(lines_file, line)
                else:
                    calls_before = len(recorder.calls)
                    result = pipeline.run(ref, repo_root, client=recorder, config=filtered)
                    cost_usd, seconds = _result_cost(result)
                    spent += cost_usd
                    done.add((task.id, version))
                    _append_jsonl(lines_file, scan_line(task.id, version, result, cost_usd, seconds))
                    if split == "train" and result.state is State.PROVEN:
                        proof_calls = [call for call in recorder.calls[calls_before:] if call.step == "proof"]
                        if proof_calls:
                            _append_jsonl(proofs_file, proof_line(task.id, proof_calls[-1]))
            if stopped_early:
                break
    lines = _read_jsonl(lines_file, "results")
    metrics = _metrics_spec_only(lines) if spec_only else _metrics_full(lines)
    return _write_run_json(
        out_dir,
        split=split,
        max_cost_usd=max_cost_usd,
        stages=stages,
        spec_model=spec_model,
        contributor=contributor,
        spec_only=spec_only,
        task_ids=task_ids,
        excluded=excluded,
        stopped_early=stopped_early,
        status="done",
        metrics=metrics,
    )


__all__ = [
    "BUGGY",
    "FIXED",
    "TaskSkipped",
    "Version",
    "apply_options",
    "check_body",
    "classify_spec",
    "entry_point_of",
    "proof_line",
    "run_harness",
    "run_spec_only",
    "scan_line",
    "solution_source",
    "task_tests_pass",
    "test_module_source",
    "write_task_repo",
]
