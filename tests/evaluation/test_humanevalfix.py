"""Tests for the HumanEvalFix harness (issue #47).

The pipeline and the spec steps are monkeypatched: these tests check the harness wiring (metrics, resume,
cost cap, spec-only flow, model overrides, exclusions) without running models or Dafny. The per-task exclusion
check runs real task tests in the sandbox, which is local and needs no network.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import pcc.context
import pcc.evaluation.__main__ as eval_main
import pcc.evaluation.datasets as datasets
import pcc.intent
import pcc.models
import pcc.pipeline
import pcc.spec
from pcc.config import Config
from pcc.core import (
    CheckKind,
    CheckRecord,
    Claim,
    Context,
    Counterexample,
    Evidence,
    FunctionRef,
    Intent,
    ProofOutcome,
    Result,
    SourceKind,
    Spec,
    State,
    ToolError,
    Translation,
    TranslationPath,
    Usage,
    source_hash,
)
from pcc.evaluation.datasets import Task
from pcc.evaluation.humanevalfix import run_harness
from pcc.models import FakeClient, Message, ModelClient, Router
from pcc.models.client import Completion

OUTCOMES = ("pass", "false_alarm", "caught", "missed", "not_checkable")


def make_task(
    number: int,
    entry_point: str,
    declaration: str,
    fixed_body: str,
    buggy_body: str,
    test: str,
) -> Task:
    """One HumanEvalFix-style task with fixed and buggy solutions and a task test."""
    payload = {
        "task_id": f"Python/{number}",
        "declaration": declaration,
        "canonical_solution": fixed_body,
        "buggy_solution": buggy_body,
        "entry_point": entry_point,
        "test": test,
        "import": "",
        "test_setup": "",
    }
    return Task(
        id=f"humanevalfix/Python/{number}",
        source_task=f"humaneval/HumanEval/{number}",
        payload=payload,
    )


def add_task() -> Task:
    return make_task(
        0,
        "add",
        "def add(a, b):",
        "    return a + b",
        "    return a - b",
        "def check(add):\n    assert add(1, 2) == 3\ncheck(add)",
    )


def neg_task() -> Task:
    return make_task(
        1,
        "neg",
        "def neg(x):",
        "    return -x",
        "    return x",
        "def check(neg):\n    assert neg(3) == -3\n    assert neg(0) == 0\ncheck(neg)",
    )


def function_ref_for(source: str, entry_point: str) -> FunctionRef:
    return FunctionRef("solution.py", entry_point, 1, source, source_hash(source))


def proven_result(function: FunctionRef, cost: float, path: TranslationPath = TranslationPath.TRANSPILER) -> Result:
    return Result(
        function,
        State.PROVEN,
        "",
        None,
        None,
        Translation(function, path, path, "dafny fn", "dafny spec"),
        (),
        None,
        ProofOutcome(True, "module M {}", 1, "spark"),
        None,
        (Usage("proof", "muse-spark-1.3", 100, 50, cost, 2.0),),
    )


def counterexample_result(function: FunctionRef, cost: float) -> Result:
    return Result(
        function,
        State.COUNTEREXAMPLE,
        "spec fails",
        None,
        None,
        Translation(function, TranslationPath.TRANSPILER, TranslationPath.TRANSPILER, "dafny fn", "dafny spec"),
        (),
        Counterexample("1, 2", "the sum", "1", "def test_add():\n    assert add(1, 2) == 3"),
        None,
        None,
        (Usage("proof", "muse-spark-1.3", 100, 50, cost, 1.0),),
    )


class ScriptedPipeline:
    """A fake ``pipeline.run`` that answers from the solution source and records configs."""

    def __init__(self, behavior: Callable[[FunctionRef, str], Result]) -> None:
        self._behavior = behavior
        self.configs: list[Config] = []
        self.calls: list[str] = []

    def __call__(
        self,
        function: FunctionRef,
        repo_root: Path,
        *,
        client: ModelClient,
        config: Config,
        callees: Mapping[FunctionRef, Result] | None = None,
    ) -> Result:
        self.configs.append(config)
        source = (repo_root / "solution.py").read_text(encoding="utf-8")
        self.calls.append(source)
        return self._behavior(function, source)


def metrics_behavior(function: FunctionRef, source: str) -> Result:
    """Fixed versions prove, buggy ``add`` is caught, buggy ``neg`` is missed as proven."""
    if "a - b" in source:
        return counterexample_result(function, 0.5)
    if "return x" in source and "return -x" not in source:
        return proven_result(function, 0.5)
    path = TranslationPath.AI if "neg" in source else TranslationPath.TRANSPILER
    return proven_result(function, 0.5, path)


def patch_pipeline(monkeypatch: pytest.MonkeyPatch, behavior: Callable[[FunctionRef, str], Result]) -> ScriptedPipeline:
    fake = ScriptedPipeline(behavior)
    monkeypatch.setattr(pcc.pipeline, "run", fake)
    return fake


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_metrics_from_two_tasks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "first"
    fake = patch_pipeline(monkeypatch, metrics_behavior)
    summary = run_harness(
        split="test", out_dir=out, max_cost_usd=10.0, tasks=(add_task(), neg_task()), client=FakeClient([])
    )
    assert len(fake.calls) == 4
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["harness"] == "humanevalfix"
    assert run["split"] == "test"
    assert run["spec_only"] is False
    assert run["stopped_early"] is False
    assert run["excluded"] == []
    metrics = run["metrics"]
    assert metrics["proof_rate_fixed"] == pytest.approx(1.0)
    assert metrics["bugs_found"] == 1
    assert metrics["false_alarms"] == 0
    assert metrics["transpiler_share"] == pytest.approx(0.75)
    assert metrics["total_cost_usd"] == pytest.approx(2.0)
    assert summary["metrics"] == metrics
    scan = read_jsonl(out / "scan.jsonl")
    assert len(scan) == 4
    assert [line["project"] for line in scan] == [
        "humanevalfix-fixed",
        "humanevalfix-buggy",
        "humanevalfix-fixed",
        "humanevalfix-buggy",
    ]
    assert [line["task_id"] for line in scan] == [
        "humanevalfix/Python/0",
        "humanevalfix/Python/0",
        "humanevalfix/Python/1",
        "humanevalfix/Python/1",
    ]
    assert [line["state"] for line in scan] == ["proven", "counterexample", "proven", "proven"]
    assert all(line["cost_usd"] == pytest.approx(0.5) for line in scan)


def test_resume_skips_done_tasks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "resume"
    out.mkdir(parents=True)
    done = {
        "project": "humanevalfix-fixed",
        "task_id": "humanevalfix/Python/0",
        "version": "fixed",
        "state": "proven",
        "reason": "",
        "name": "add",
        "path": "solution.py",
        "cost_usd": 0.5,
        "seconds": 2.0,
        "result": {},
    }
    (out / "scan.jsonl").write_text(json.dumps(done) + "\n", encoding="utf-8")
    (out / "scan.jsonl").write_text(
        (out / "scan.jsonl").read_text(encoding="utf-8")
        + json.dumps({**done, "project": "humanevalfix-buggy", "version": "buggy", "state": "counterexample"})
        + "\n",
        encoding="utf-8",
    )
    fake = patch_pipeline(monkeypatch, metrics_behavior)
    run_harness(split="test", out_dir=out, max_cost_usd=10.0, tasks=(add_task(), neg_task()), client=FakeClient([]))
    assert all("neg" in source for source in fake.calls)
    assert len(fake.calls) == 2
    scan = read_jsonl(out / "scan.jsonl")
    assert len(scan) == 4


def test_resume_rejects_option_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "mismatch"
    patch_pipeline(monkeypatch, metrics_behavior)
    run_harness(split="test", out_dir=out, max_cost_usd=10.0, tasks=(add_task(),), client=FakeClient([]))
    with pytest.raises(ToolError, match="stages mismatch"):
        run_harness(
            split="test",
            out_dir=out,
            max_cost_usd=10.0,
            stages="spark",
            tasks=(add_task(),),
            client=FakeClient([]),
        )
    with pytest.raises(ToolError, match="spec_model mismatch"):
        run_harness(
            split="test",
            out_dir=out,
            max_cost_usd=10.0,
            spec_model="glimmer",
            tasks=(add_task(),),
            client=FakeClient([]),
        )
    with pytest.raises(ToolError, match="max_cost_usd mismatch"):
        run_harness(split="test", out_dir=out, max_cost_usd=5.0, tasks=(add_task(),), client=FakeClient([]))


def test_cost_cap_stops_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "capped"
    fake = patch_pipeline(monkeypatch, metrics_behavior)
    run_harness(split="test", out_dir=out, max_cost_usd=0.5, tasks=(add_task(), neg_task()), client=FakeClient([]))
    assert len(fake.calls) == 1
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["stopped_early"] is True
    assert len(read_jsonl(out / "scan.jsonl")) == 1


def spec_only_fakes(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """Monkeypatch steps 1 to 3: fixed specs pass, buggy specs fail the original check."""
    client = FakeClient([])

    def fake_collect(function: FunctionRef, repo_root: Path) -> Context:
        return Context(function, ())

    def fake_derive(context: Context, *, client: ModelClient, model: str) -> Intent:
        evidence = Evidence(SourceKind.DOCSTRING, "solution.py:1", "adds two numbers")
        return Intent(context.function, "adds two numbers", (Claim("adds two numbers", (evidence,)),))

    def fake_write(intent: Intent, *, client: ModelClient, model: str, hint: str = "") -> Spec:
        return Spec(intent.function, "def spec(a, b, result):\n    return True", None, ())

    def fake_check(
        spec: Spec,
        function: FunctionRef,
        repo_root: Path,
        *,
        client: ModelClient,
        model: str,
        config: Any,
    ) -> Spec:
        buggy = "a - b" in function.source or ("return x" in function.source and "return -x" not in function.source)
        hyp = CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, not buggy, "" if not buggy else "add(1, 2) == 1")
        mutants = CheckRecord(CheckKind.MUTANTS, True, "")
        return Spec(spec.function, spec.predicate, spec.precondition, (hyp, mutants))

    def no_pipeline(*args: Any, **kwargs: Any) -> Result:
        raise AssertionError("pipeline.run must not be called with --spec-only")

    monkeypatch.setattr(pcc.context, "collect", fake_collect)
    monkeypatch.setattr(pcc.intent, "derive", fake_derive)
    monkeypatch.setattr(pcc.spec, "write", fake_write)
    monkeypatch.setattr(pcc.spec, "check", fake_check)
    monkeypatch.setattr(pcc.pipeline, "run", no_pipeline)
    return client


def test_spec_only_outcomes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "speconly"
    client = spec_only_fakes(monkeypatch)
    summary = run_harness(
        split="test", out_dir=out, max_cost_usd=10.0, spec_only=True, tasks=(add_task(),), client=client
    )
    assert client.calls == []
    lines = read_jsonl(out / "spec_only.jsonl")
    assert [(line["task_id"], line["version"], line["outcome"]) for line in lines] == [
        ("humanevalfix/Python/0", "fixed", "pass"),
        ("humanevalfix/Python/0", "buggy", "caught"),
    ]
    assert all(set(line) == {"task_id", "version", "outcome", "reason", "cost_usd", "seconds"} for line in lines)
    assert not (out / "scan.jsonl").exists()
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["spec_only"] is True
    assert run["metrics"]["outcomes"] == {"pass": 1, "false_alarm": 0, "caught": 1, "missed": 0, "not_checkable": 0}
    assert summary["metrics"] == run["metrics"]


def test_spec_model_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "specmodel"
    fake = patch_pipeline(monkeypatch, metrics_behavior)
    run_harness(
        split="test",
        out_dir=out,
        max_cost_usd=10.0,
        spec_model="glimmer",
        tasks=(add_task(),),
        client=FakeClient([]),
    )
    assert fake.configs
    assert all(config.models.spec == "glimmer" for config in fake.configs)


def test_excluded_when_fixed_fails_own_tests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    broken = make_task(
        2,
        "add",
        "def add(a, b):",
        "    return a - b",
        "    return a - b",
        "def check(add):\n    assert add(1, 2) == 3\ncheck(add)",
    )
    out = tmp_path / "runs" / "excluded"
    fake = patch_pipeline(monkeypatch, metrics_behavior)
    run_harness(split="test", out_dir=out, max_cost_usd=10.0, tasks=(broken, add_task()), client=FakeClient([]))
    assert len(fake.calls) == 2
    assert "a + b" in fake.calls[0]
    assert "a - b" in fake.calls[1]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["excluded"] == [{"task_id": "humanevalfix/Python/2", "reason": "fixed version fails its own tests"}]
    assert run["metrics"]["proof_rate_fixed"] == pytest.approx(1.0)


def test_proofs_jsonl_on_train_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "train"

    def behavior(function: FunctionRef, source: str) -> Result:
        return proven_result(function, 0.1)

    fake = patch_pipeline(monkeypatch, behavior)
    answers = ['{"ok": true}', '{"ok": true}', '{"ok": true}', '{"ok": true}']
    client = FakeClient(answers)
    original = fake.__call__

    def recording_call(
        function: FunctionRef,
        repo_root: Path,
        *,
        client: ModelClient,
        config: Config,
        callees: Mapping[FunctionRef, Result] | None = None,
    ) -> Result:
        client.complete(
            config.models.spec, (Message("user", "prove it"),), step="draft", temperature=0.8, max_tokens=10
        )
        answer = client.complete(
            config.models.spec, (Message("user", "prove it"),), step="proof", temperature=0.8, max_tokens=10
        )
        assert isinstance(answer, Completion)
        return original(function, repo_root, client=client, config=config, callees=callees)

    monkeypatch.setattr(pcc.pipeline, "run", recording_call)
    run_harness(split="train", out_dir=out, max_cost_usd=10.0, tasks=(add_task(),), client=client)
    lines = read_jsonl(out / "proofs.jsonl")
    assert len(lines) == 2
    assert all(line["task_id"] == "humanevalfix/Python/0" for line in lines)
    assert all(line["messages"] == [{"role": "user", "content": "prove it"}] for line in lines)
    assert all(line["answer"] == '{"ok": true}' for line in lines)


def test_no_proofs_jsonl_on_test_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "testsplit"
    patch_pipeline(monkeypatch, metrics_behavior)
    run_harness(split="test", out_dir=out, max_cost_usd=10.0, tasks=(add_task(),), client=FakeClient([]))
    assert not (out / "proofs.jsonl").exists()


def test_unknown_stage_exits_with_usage_error(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "badstage"
    assert (
        eval_main.main(
            ["humanevalfix", "--split", "test", "--out", str(out), "--max-cost-usd", "1", "--stages", "nope"]
        )
        == 2
    )


def test_tool_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken_split(
        name: str, which: str, cache_dir: Path | None = None, splits_dir: Path | None = None
    ) -> tuple[Task, ...]:
        raise ToolError("cache is broken")

    monkeypatch.setattr(datasets, "split", broken_split)
    out = tmp_path / "runs" / "broken"
    assert eval_main.main(["humanevalfix", "--split", "test", "--out", str(out), "--max-cost-usd", "1"]) == 1


def test_main_writes_run_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_split(
        name: str, which: str, cache_dir: Path | None = None, splits_dir: Path | None = None
    ) -> tuple[Task, ...]:
        assert name == "humanevalfix"
        return (add_task(),)

    client = FakeClient([])
    router = Router(
        {"meta": client, "glimmer": client},
        {"muse-spark-1.3": "meta", "muse-wonder-1.0": "meta"},
        {},
        {"META_API_KEY": "test", "PCC_META_ALLOWED": "1"},
    )

    def fake_build_client(config: Config, env: object = None, **kwargs: object) -> Router:
        return router

    monkeypatch.setattr(datasets, "split", fake_split)
    monkeypatch.setattr(pcc.models, "build_client", fake_build_client)
    patch_pipeline(monkeypatch, metrics_behavior)
    out = tmp_path / "runs" / "cli"
    assert eval_main.main(["humanevalfix", "--split", "test", "--out", str(out), "--max-cost-usd", "10"]) == 0
    assert (out / "run.json").is_file()
    assert len(read_jsonl(out / "scan.jsonl")) == 2
