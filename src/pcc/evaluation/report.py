"""Published evaluation reports (docs/plan.md, "Evaluation", "Protocol").

A raw run folder ``runs/<name>`` holds ``run.json`` (written by the harness) plus one results file: the
Vericoding harness writes ``results.jsonl`` with one line per task (``task_id``, ``attempts``,
``verified_attempts``, ``cost_usd``, ``seconds``), the HumanEvalFix pipeline writes ``scan.jsonl`` with one
line per evaluated version (``task_id``, ``version``, ``state``, ``cost_usd``, ``seconds``, ``result``), and
the spec writer run writes ``spec_only.jsonl`` with one line per evaluated version (``task_id``, ``version``,
``outcome``, ``cost_usd``, ``seconds``). ``run.json`` records the run metadata, including the UTC ``date``
and the ``commit`` (git SHA) the run was recorded with; the folder name and the history row are built from
those fields, never from the current checkout.

``publish`` turns one run into ``docs/results/<date>-<short sha>-<name>/`` with ``meta.json`` (the fields of
``run.json``), ``summary.json`` (metrics, the main rate with its Wilson interval, and the solved task IDs)
and ``summary.md``, and appends one row to ``docs/results/history.md``. ``compare`` builds one markdown
table over published runs; comparisons are paired, so every run must cover the same task IDs.

Run as ``python -m pcc.evaluation.report``; usage errors exit 2. No network, no model calls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HISTORY_HEADER = "| date | commit | model | dataset | n | rate | cost |"

_RESULTS_FILE = {"vericoding": "results.jsonl", "pipeline": "scan.jsonl", "spec": "spec_only.jsonl"}

_STATES = ("proven", "counterexample", "open", "not_checkable")

_SPEC_OUTCOMES = ("pass", "false_alarm", "caught", "missed", "not_checkable")


class UsageError(ValueError):
    """A usage error: the command exits 2 with this message."""


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for ``successes`` of ``n`` trials at ``z`` standard deviations.

    Returns ``(0.0, 0.0)`` for ``n == 0``. The default ``z = 1.96`` is the 95 % interval.
    """
    if n == 0:
        return (0.0, 0.0)
    if successes < 0 or n < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got {successes} of {n}")
    if z <= 0:
        raise ValueError(f"need z > 0, got {z}")
    center = successes / n + z * z / (2 * n)
    spread = z * math.sqrt(successes * (n - successes) / (n * n * n) + z * z / (4 * n * n))
    width = 1 + z * z / n
    return (max(0.0, (center - spread) / width), min(1.0, (center + spread) / width))


def _read_json(path: Path, label: str) -> Any:
    """The parsed JSON document of ``path``; a missing or broken file is a usage error."""
    if not path.is_file():
        raise UsageError(f"{label} {path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise UsageError(f"{label} {path} is not valid JSON: {exc}") from exc


def _result_lines(path: Path) -> list[dict[str, Any]]:
    """The parsed JSON lines of a results file; a missing file is an empty run."""
    if not path.is_file():
        return []
    try:
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except ValueError as exc:
        raise UsageError(f"results file {path} is not valid JSON lines: {exc}") from exc
    lines: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise UsageError(f"results file {path} holds a line that is not a JSON object")
        lines.append(entry)
    return lines


def _task_id(line: dict[str, Any], path: Path) -> str:
    task_id = line.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise UsageError(f"results file {path} holds a line without a task_id")
    return task_id


def _number(line: dict[str, Any], key: str, path: Path) -> float:
    value = line.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageError(f"results file {path} holds a non-numeric {key!r}: {value!r}")
    return float(value)


def _count(line: dict[str, Any], key: str, path: Path) -> int:
    value = line.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise UsageError(f"results file {path} holds a non-integer {key!r}: {value!r}")
    if value < 0:
        raise UsageError(f"results file {path} holds a negative {key!r}: {value!r}")
    return value


def _kind_of(run: dict[str, Any]) -> str:
    """The run kind from ``run.json``: ``pipeline``, ``spec`` or ``vericoding``."""
    if run.get("harness") == "humanevalfix":
        return "spec" if run.get("spec_only") is True else "pipeline"
    return "vericoding"


def _metadata(run: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """The (date, commit, model, mode, dataset) of a run; a missing date or commit is a usage error."""
    date = run.get("date")
    if not isinstance(date, str) or not date.strip():
        raise UsageError("run.json has no 'date': the harness must record the UTC date of the run")
    commit = run.get("commit")
    if not isinstance(commit, str) or not commit.strip():
        raise UsageError("run.json has no 'commit': the harness must record the git SHA of the run")
    model = run.get("model")
    if not isinstance(model, str) or not model:
        spec_model = run.get("spec_model")
        model = spec_model if isinstance(spec_model, str) and spec_model else "unknown"
    mode = "contributor" if run.get("contributor", False) else "standard"
    dataset = run.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        harness = run.get("harness")
        split = run.get("split")
        if isinstance(harness, str) and harness and isinstance(split, str) and split:
            dataset = f"{harness}/{split}"
        elif isinstance(harness, str) and harness:
            dataset = harness
        else:
            dataset = "unknown"
    return date, commit, model, mode, dataset


def _vericoding_metrics(
    lines: list[dict[str, Any]], path: Path, stopped: bool
) -> tuple[dict[str, Any], list[str], str, float, float, float, int, float]:
    """Metrics, solved task IDs, rate name, rate, interval, n and total cost of a Vericoding run."""
    solved: list[str] = []
    terms: list[float] = []
    cost_total = 0.0
    seconds_total = 0.0
    for line in lines:
        task_id = _task_id(line, path)
        attempts = _count(line, "attempts", path)
        verified = _count(line, "verified_attempts", path)
        if verified > attempts:
            raise UsageError(f"results file {path} holds more verified than attempted for {task_id}")
        cost_total += _number(line, "cost_usd", path)
        seconds_total += _number(line, "seconds", path)
        terms.append(verified / attempts if attempts else 0.0)
        if verified >= 1:
            solved.append(task_id)
    n = len(lines)
    rate = len(solved) / n if n else 0.0
    low, high = wilson(len(solved), n)
    metrics = {
        "pass_at_1": sum(terms) / n if n else 0.0,
        "pass_at_k": rate,
        "interval": [low, high],
        "cost_usd_total": cost_total,
        "seconds_per_task": seconds_total / n if n else 0.0,
        "stopped_at_cost_cap": stopped,
    }
    return metrics, sorted(set(solved)), "pass@k", rate, low, high, n, cost_total


def _proof_stage(line: dict[str, Any]) -> str | None:
    """The prover stage behind a proven scan line, or None when the line is not proven."""
    if line.get("state") != "proven":
        return None
    result = line.get("result")
    if not isinstance(result, dict):
        return "unknown"
    proof = result.get("proof")
    if not isinstance(proof, dict) or not proof.get("proven", False):
        return "unknown"
    stage = proof.get("stage")
    return stage if isinstance(stage, str) and stage else "unknown"


def _pipeline_metrics(
    lines: list[dict[str, Any]], path: Path, stopped: bool
) -> tuple[dict[str, Any], list[str], str, float, float, float, int, float]:
    """Metrics, solved task IDs, rate name, rate, interval, n and total cost of a pipeline run."""
    for line in lines:
        version = line.get("version")
        if version not in ("fixed", "buggy"):
            raise UsageError(f"results file {path} holds an unknown version: {version!r}")
        if line.get("state") not in _STATES:
            raise UsageError(f"results file {path} holds an unknown state: {line.get('state')!r}")
    fixed = [line for line in lines if line.get("version") == "fixed"]
    buggy = [line for line in lines if line.get("version") == "buggy"]
    proven_fixed = sum(1 for line in fixed if line.get("state") == "proven")
    rate = proven_fixed / len(fixed) if fixed else 0.0
    low, high = wilson(proven_fixed, len(fixed))

    def label(line: dict[str, Any]) -> str:
        task_id = line.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
        name = line.get("name")
        return name if isinstance(name, str) and name else "unknown"

    false_proofs = sorted(label(line) for line in buggy if line.get("state") == "proven")
    state_counts = {state: sum(1 for line in lines if line.get("state") == state) for state in _STATES}
    proven = [line for line in lines if line.get("state") == "proven"]
    stage_counts: dict[str, int] = {}
    for line in proven:
        stage = _proof_stage(line)
        assert stage is not None
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    stages = {stage: {"count": count, "share": count / len(proven)} for stage, count in sorted(stage_counts.items())}

    def translation_path(line: dict[str, Any]) -> str | None:
        result = line.get("result")
        if not isinstance(result, dict):
            return None
        translation = result.get("translation")
        if not isinstance(translation, dict):
            return None
        path = translation.get("function_path")
        return path if isinstance(path, str) else None

    translated = [line for line in lines if translation_path(line) is not None]
    transpiled = sum(1 for line in translated if translation_path(line) == "transpiler")
    cost_total = sum(_number(line, "cost_usd", path) for line in lines)
    seconds_total = sum(_number(line, "seconds", path) for line in lines)
    metrics = {
        "proof_rate_fixed": rate,
        "interval": [low, high],
        "n_fixed": len(fixed),
        "n_buggy": len(buggy),
        "bugs_found": sum(1 for line in buggy if line.get("state") == "counterexample"),
        "false_alarms": sum(1 for line in fixed if line.get("state") == "counterexample"),
        "false_proofs": false_proofs,
        "state_counts": state_counts,
        "state_shares": {state: count / len(lines) if lines else 0.0 for state, count in state_counts.items()},
        "proof_stages": stages,
        "transpiler_share": transpiled / len(translated) if translated else None,
        "cost_usd_total": cost_total,
        "mean_cost_usd_per_function": cost_total / len(lines) if lines else 0.0,
        "mean_seconds_per_function": seconds_total / len(lines) if lines else 0.0,
        "stopped_at_cost_cap": stopped,
    }
    solved = sorted({_task_id(line, path) for line in fixed if line.get("state") == "proven"})
    task_ids = sorted({_task_id(line, path) for line in lines})
    return metrics, solved, "proof_rate_fixed", rate, low, high, len(task_ids), cost_total


def _spec_metrics(
    lines: list[dict[str, Any]], path: Path, stopped: bool
) -> tuple[dict[str, Any], list[str], str, float, float, float, int, float]:
    """Metrics, solved task IDs, rate name, rate, interval, n and total cost of a spec writer run."""
    for line in lines:
        if line.get("version") not in ("fixed", "buggy"):
            raise UsageError(f"results file {path} holds an unknown version: {line.get('version')!r}")
        if line.get("outcome") not in _SPEC_OUTCOMES:
            raise UsageError(f"results file {path} holds an unknown outcome: {line.get('outcome')!r}")
    fixed = [line for line in lines if line.get("version") == "fixed"]
    buggy = [line for line in lines if line.get("version") == "buggy"]
    passes = sum(1 for line in fixed if line.get("outcome") == "pass")
    caught = sum(1 for line in buggy if line.get("outcome") == "caught")
    false_alarms = sum(1 for line in fixed if line.get("outcome") == "false_alarm")
    not_checkable = sum(1 for line in lines if line.get("outcome") == "not_checkable")
    correct = passes + caught
    rate = correct / len(lines) if lines else 0.0
    low, high = wilson(correct, len(lines))
    pass_low, pass_high = wilson(passes, len(fixed))
    caught_low, caught_high = wilson(caught, len(buggy))
    alarm_low, alarm_high = wilson(false_alarms, len(fixed))
    check_low, check_high = wilson(not_checkable, len(lines))
    cost_total = sum(_number(line, "cost_usd", path) for line in lines)
    seconds_total = sum(_number(line, "seconds", path) for line in lines)
    metrics = {
        "correct_passes": passes,
        "correct_passes_interval": [pass_low, pass_high],
        "n_fixed": len(fixed),
        "buggy_caught": caught,
        "buggy_caught_interval": [caught_low, caught_high],
        "n_buggy": len(buggy),
        "false_alarms": false_alarms,
        "false_alarms_interval": [alarm_low, alarm_high],
        "missed": sum(1 for line in buggy if line.get("outcome") == "missed"),
        "not_checkable": not_checkable,
        "not_checkable_interval": [check_low, check_high],
        "correct_total": correct,
        "evaluated": len(lines),
        "cost_usd_total": cost_total,
        "mean_cost_usd_per_function": cost_total / len(lines) if lines else 0.0,
        "mean_seconds_per_function": seconds_total / len(lines) if lines else 0.0,
        "stopped_at_cost_cap": stopped,
    }
    by_task: dict[str, dict[str, str]] = {}
    for line in lines:
        version = line.get("version")
        if version in ("fixed", "buggy"):
            by_task.setdefault(_task_id(line, path), {})[version] = str(line.get("outcome"))
    solved = sorted(
        task
        for task, outcomes in by_task.items()
        if outcomes.get("fixed") == "pass" and outcomes.get("buggy") == "caught"
    )
    task_ids = sorted({_task_id(line, path) for line in lines})
    return metrics, solved, "correct_rate", rate, low, high, len(task_ids), cost_total


def _format_rate(rate: float, low: float, high: float, name: str) -> str:
    return f"{name} {rate:.3f} (95% CI {low:.3f}-{high:.3f})"


def _summary_markdown(
    *,
    name: str,
    kind: str,
    model: str,
    mode: str,
    dataset: str,
    date: str,
    commit: str,
    rate_text: str,
    metrics: dict[str, Any],
    stopped: bool,
) -> str:
    """The human-readable ``summary.md``; every results table states the contributor mode."""
    if mode == "contributor":
        mode_line = "- Mode: contributor (contributor mode on; the content may be used to train models)"
    else:
        mode_line = "- Mode: standard (contributor mode off; the content is not used to train models)"
    lines = [
        f"# Evaluation report: {name}",
        "",
        f"- Kind: {kind}",
        f"- Model: {model}",
        f"- Dataset: {dataset}",
        mode_line,
        f"- Date (UTC): {date}",
        f"- Commit: {commit}",
        f"- Cost cap reached: {'yes' if stopped else 'no'}",
        "",
        "## Metrics",
        "",
        f"- Main rate: {rate_text}",
    ]
    if kind == "vericoding":
        lines += [
            f"- pass@1: {metrics['pass_at_1']:.3f}",
            f"- Total cost: {metrics['cost_usd_total']:.2f} USD",
            f"- Seconds per task: {metrics['seconds_per_task']:.1f}",
        ]
    elif kind == "pipeline":
        lines += [
            f"- Fixed versions (n): {metrics['n_fixed']}",
            f"- Buggy versions (n): {metrics['n_buggy']}",
            f"- Bugs found: {metrics['bugs_found']}",
            f"- False alarms: {metrics['false_alarms']}",
            f"- False proofs: {len(metrics['false_proofs'])}",
        ]
        for function in metrics["false_proofs"]:
            lines.append(f"  - {function}")
        lines.append("- Share per final state:")
        for state in _STATES:
            lines.append(f"  - {state}: {metrics['state_counts'][state]} ({metrics['state_shares'][state]:.3f})")
        lines.append("- Share per proof stage (of proven functions):")
        for stage, share in metrics["proof_stages"].items():
            lines.append(f"  - {stage}: {share['count']} ({share['share']:.3f})")
        if metrics["transpiler_share"] is None:
            lines.append("- Transpiler share: unknown (no translation info in scan.jsonl)")
        else:
            lines.append(f"- Transpiler share: {metrics['transpiler_share']:.3f}")
        lines += [
            f"- Mean cost per function: {metrics['mean_cost_usd_per_function']:.2f} USD",
            f"- Mean time per function: {metrics['mean_seconds_per_function']:.1f} s",
        ]
    else:
        lines += [
            f"- Correct passes: {metrics['correct_passes']} of {metrics['n_fixed']} "
            f"(95% CI {metrics['correct_passes_interval'][0]:.3f}-{metrics['correct_passes_interval'][1]:.3f})",
            f"- Buggy functions caught: {metrics['buggy_caught']} of {metrics['n_buggy']} "
            f"(95% CI {metrics['buggy_caught_interval'][0]:.3f}-{metrics['buggy_caught_interval'][1]:.3f})",
            f"- False alarms: {metrics['false_alarms']} "
            f"(95% CI {metrics['false_alarms_interval'][0]:.3f}-{metrics['false_alarms_interval'][1]:.3f})",
            f"- Not checkable: {metrics['not_checkable']} "
            f"(95% CI {metrics['not_checkable_interval'][0]:.3f}-{metrics['not_checkable_interval'][1]:.3f})",
            f"- Mean cost per function: {metrics['mean_cost_usd_per_function']:.2f} USD",
            f"- Mean time per function: {metrics['mean_seconds_per_function']:.1f} s",
        ]
    return "\n".join(lines) + "\n"


def publish(run_dir: str | Path, results_dir: str | Path = "docs/results") -> Path:
    """Publish the raw run in ``run_dir``; return the published folder.

    Reads ``run.json`` and the run's results file, writes ``<date>-<short sha>-<name>/`` with ``meta.json``,
    ``summary.json`` and ``summary.md`` under ``results_dir``, and appends one row to ``history.md``. A run
    without ``run.json``, without a date or commit in ``run.json``, or whose folder already exists is a
    usage error; an empty run publishes ``n = 0``.
    """
    run_path = Path(run_dir)
    results_path = Path(results_dir)
    run_file = run_path / "run.json"
    if not run_file.is_file():
        raise UsageError(f"run {run_path} has no run.json")
    run = _read_json(run_file, "run.json")
    if not isinstance(run, dict):
        raise UsageError(f"run.json of {run_path} must hold a JSON object")
    date, commit, model, mode, dataset = _metadata(run)
    kind = _kind_of(run)
    name = run_path.name
    dest = results_path / f"{date}-{commit[:7]}-{name}"
    if dest.exists():
        raise UsageError(f"published folder {dest} already exists; not overwriting it")
    lines = _result_lines(run_path / _RESULTS_FILE[kind])
    stopped = run.get("stopped_early") is True
    if kind == "pipeline":
        metrics, solved, rate_name, rate, low, high, n, cost = _pipeline_metrics(lines, run_path, stopped)
        task_ids = sorted({_task_id(line, run_path) for line in lines})
    elif kind == "spec":
        metrics, solved, rate_name, rate, low, high, n, cost = _spec_metrics(lines, run_path, stopped)
        task_ids = sorted({_task_id(line, run_path) for line in lines})
    else:
        metrics, solved, rate_name, rate, low, high, n, cost = _vericoding_metrics(lines, run_path, stopped)
        task_ids = sorted({_task_id(line, run_path) for line in lines})
    rate_text = _format_rate(rate, low, high, rate_name)
    summary: dict[str, Any] = {
        "run": name,
        "kind": kind,
        "model": model,
        "mode": mode,
        "dataset": dataset,
        "date": date,
        "commit": commit,
        "n": n,
        "rate": rate,
        "rate_lo": low,
        "rate_hi": high,
        "rate_name": rate_name,
        "cost_usd": cost,
        "task_ids": task_ids,
        "solved_task_ids": solved,
        "metrics": metrics,
    }
    dest.mkdir(parents=True)
    (dest / "meta.json").write_text(json.dumps(run, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (dest / "summary.md").write_text(
        _summary_markdown(
            name=name,
            kind=kind,
            model=model,
            mode=mode,
            dataset=dataset,
            date=date,
            commit=commit,
            rate_text=rate_text,
            metrics=metrics,
            stopped=stopped,
        ),
        encoding="utf-8",
    )
    history = results_path / "history.md"
    row = f"| {date} | {commit[:7]} | {model} | {dataset} | {n} | {rate:.3f} | {cost:.2f} |"
    if not history.is_file() or not history.read_text(encoding="utf-8").strip():
        history.write_text(HISTORY_HEADER + "\n" + row + "\n", encoding="utf-8")
    else:
        with history.open("a", encoding="utf-8") as handle:
            handle.write(row + "\n")
    return dest


def _load_published(folder: Path) -> tuple[str, dict[str, Any]]:
    """The (name, summary) of a published run folder; a missing ``summary.json`` is a usage error."""
    if not folder.is_dir():
        raise UsageError(f"published run {folder} does not exist")
    summary = _read_json(folder / "summary.json", "summary.json")
    if not isinstance(summary, dict):
        raise UsageError(f"summary.json of {folder} must hold a JSON object")
    required = (
        "model",
        "mode",
        "dataset",
        "n",
        "rate",
        "rate_lo",
        "rate_hi",
        "rate_name",
        "cost_usd",
        "task_ids",
        "solved_task_ids",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise UsageError(f"summary.json of {folder} misses {', '.join(missing)}: not a published run")
    return folder.name, summary


def compare(folders: Sequence[str | Path], out: str | Path | None = None) -> str:
    """Compare published runs: one markdown table plus the paired task differences.

    Every run must cover the same task IDs, otherwise this is a usage error naming the missing IDs. When
    ``out`` is given the table is written there too. Returns the markdown.
    """
    runs = [_load_published(Path(folder)) for folder in folders]
    if not runs:
        raise UsageError("compare needs at least one published run")
    union = sorted({task for _, summary in runs for task in summary["task_ids"]})
    problems = []
    for name, summary in runs:
        missing = [task for task in union if task not in summary["task_ids"]]
        if missing:
            problems.append(f"run '{name}' misses task IDs: {', '.join(missing)}")
    if problems:
        raise UsageError("runs cover different task IDs; " + "; ".join(problems))
    lines = [
        "# Run comparison",
        "",
        "| run | model | mode | dataset | n | rate | cost (USD) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, summary in runs:
        rate = _format_rate(summary["rate"], summary["rate_lo"], summary["rate_hi"], summary["rate_name"])
        lines.append(
            f"| {name} | {summary['model']} | {summary['mode']} | {summary['dataset']} "
            f"| {summary['n']} | {rate} | {summary['cost_usd']:.2f} |"
        )
    lines += ["", "## Paired differences", ""]
    pairs = 0
    for index, (first_name, first) in enumerate(runs):
        for second_name, second in runs[index + 1 :]:
            pairs += 1
            only_first = sorted(set(first["solved_task_ids"]) - set(second["solved_task_ids"]))
            only_second = sorted(set(second["solved_task_ids"]) - set(first["solved_task_ids"]))
            lines.append(f"### {first_name} vs {second_name}")
            lines.append("")
            if not only_first and not only_second:
                lines.append(f"- Both solve the same tasks ({len(first['solved_task_ids'])} of {len(union)}).")
            else:
                if only_first:
                    lines.append(f"- Only in {first_name} ({len(only_first)}): {', '.join(only_first)}")
                if only_second:
                    lines.append(f"- Only in {second_name} ({len(only_second)}): {', '.join(only_second)}")
            lines.append("")
    if pairs == 0:
        lines.append("Only one run: no pairs to compare.")
        lines.append("")
    table = "\n".join(lines)
    if out is not None:
        out_path = Path(out)
        if out_path.parent != Path():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table, encoding="utf-8")
    return table


def build_parser() -> argparse.ArgumentParser:
    """The command line: ``publish`` one raw run, or ``compare`` published runs."""
    parser = argparse.ArgumentParser(prog="python -m pcc.evaluation.report", description="Publish evaluation runs.")
    sub = parser.add_subparsers(dest="command", required=True)
    published = sub.add_parser("publish", help="publish runs/<name> into the results directory")
    published.add_argument("run", help="raw run folder, e.g. runs/<name>")
    published.add_argument("--results-dir", default="docs/results", help="published runs go here")
    compared = sub.add_parser("compare", help="compare published runs in one markdown table")
    compared.add_argument("folders", nargs="+", help="published run folders")
    compared.add_argument("--out", default=None, help="write the markdown table to this file too")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report command named in ``argv``; return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "publish":
            publish(args.run, args.results_dir)
            return 0
        text = compare(args.folders, args.out)
        if args.out is None:
            print(text)
        return 0
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
