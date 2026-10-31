"""Pipeline: run one function and many (issue #34).

Only ``pipeline`` knows the order of the steps: every test monkeypatches the step functions and asserts the
result states, the reason strings and what ran before them. The callee gate, linking and ``run_many`` levels
come from #43 and are locked in ``tests/test_pipeline_callees.py``; this file adds the missing step-0 and
step-3-to-7 cases around that implementation.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import pcc.pipeline
from pcc.config import BudgetConfig, Config, ModelsConfig, PipelineConfig, SpecConfig
from pcc.core import (
    CheckKind,
    CheckRecord,
    Context,
    Counterexample,
    DafnyError,
    EquivalenceReport,
    Evidence,
    FunctionRef,
    Intent,
    NotCheckable,
    ProofOutcome,
    Result,
    SecondOpinion,
    SourceKind,
    Spec,
    State,
    ToolError,
    Translation,
    TranslationPath,
    UnsupportedConstruct,
)
from pcc.models import BudgetExceeded, FakeClient
from tests.helpers import contradiction as make_contradiction
from tests.helpers import counterexample as make_counterexample
from tests.helpers import function_ref
from tests.helpers import intent as make_intent
from tests.helpers import proof as make_proof
from tests.helpers import spec as make_spec

PATH = "pkg/functions.py"

SIMPLE_SOURCE = "def total(xs: list[int]) -> int:\n    return sum(xs)\n"
DAFNY_SPEC = "predicate spec(xs: seq<int>, result: int)\n{\n  result >= 0\n}\n"
DAFNY_FUNCTION = (
    "method total(xs: seq<int>) returns (result: int)\n"
    "  requires pre(xs)\n"
    "  ensures spec(xs, result)\n"
    "{\n"
    "  result := 0;\n"
    "}\n"
)


def _passing_checks() -> list[CheckRecord]:
    """A passing HYPOTHESIS_ORIGINAL plus a passing MUTANTS record."""
    return [
        CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 10, 10),
        CheckRecord(CheckKind.MUTANTS, True, ""),
    ]


def _hyp_fail(detail: str) -> list[CheckRecord]:
    """A failing HYPOTHESIS_ORIGINAL with a passing MUTANTS record."""
    return [
        CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, detail, 10, 10),
        CheckRecord(CheckKind.MUTANTS, True, ""),
    ]


class _Steps:
    """Configurable fake steps that record what ``run`` asked for.

    Tests tweak the ``plan`` attributes to script failures; every call is recorded so the tests can assert
    the step order, the models, the hints and the single second-opinion call.
    """

    def __init__(self, ref: FunctionRef) -> None:
        self.ref = ref
        self.calls: list[str] = []
        self.models: dict[str, list[str]] = {}
        self.write_hints: list[str] = []
        self.translate_mismatches: list[Counterexample | None] = []
        self.translate_count = 0
        self.search_failed_clauses: list[str | None] = []
        self.build_arguments: list[str] = []
        self.review_calls = 0
        self.review_models: list[str] = []
        self.prove_lines: list[int] = []
        self.evidence: tuple[Evidence, ...] = (Evidence(SourceKind.DOCSTRING, "doc", "the total"),)
        self.intent_value: Intent | None = None
        self.intent_raises: Exception | None = None
        self.collect_raises: Exception | None = None
        self.call_wrapped_client_in_derive = False
        self.check_plan: list[list[CheckRecord]] = []
        self.transpile_spec_plan: list[str | UnsupportedConstruct] = []
        self.transpile_plan: list[str | UnsupportedConstruct] = []
        self.equivalence_plan: list[EquivalenceReport] = []
        self.spec_check_record: CheckRecord | None = None
        self.cases: tuple[tuple[str, str], ...] = (("[]", "0"),)
        self.prove_outcome: ProofOutcome = make_proof()
        self.prove_raises: Exception | None = None
        self.search_hit: Counterexample | None = None
        self.started: dict[str, float] = {}
        self.finished: dict[str, float] = {}
        self._guard = threading.Lock()

    def _note(self, name: str, model: str | None = None) -> None:
        with self._guard:
            self.calls.append(name)
            if model is not None:
                self.models.setdefault(name, []).append(model)

    # -- step fakes ---------------------------------------------------------

    def fake_collect(self, function: FunctionRef, repo_root: Path) -> Context:
        _ = repo_root
        with self._guard:
            self.started[function.qualname] = time.monotonic()
        self._note("collect")
        if self.collect_raises is not None:
            raise self.collect_raises
        return Context(function, self.evidence)

    def fake_search_further(self, context: Context, repo_root: Path, **kwargs: Any) -> Context:
        _ = (repo_root, kwargs)
        self._note("search_further", kwargs.get("model"))
        return context

    def fake_derive(self, context: Context, **kwargs: Any) -> Intent:
        self._note("derive", kwargs.get("model"))
        if self.call_wrapped_client_in_derive:
            kwargs["client"].complete("m", (), step="intent")
        if self.intent_raises is not None:
            raise self.intent_raises
        if self.intent_value is not None:
            return self.intent_value
        return make_intent(context.function)

    def fake_write(self, intent: Intent, **kwargs: Any) -> Spec:
        _ = intent
        self._note("write", kwargs.get("model"))
        hint = kwargs.get("hint", "")
        assert isinstance(hint, str)
        self.write_hints.append(hint)
        return make_spec(intent.function)

    def fake_check(self, spec: Spec, function: FunctionRef, repo_root: Path, **kwargs: Any) -> Spec:
        _ = (function, repo_root)
        self._note("check", kwargs.get("model"))
        added = self.check_plan.pop(0) if self.check_plan else _passing_checks()
        return dataclasses.replace(spec, checks=(*spec.checks, *added))

    def fake_transpile_spec(self, spec: Spec) -> str:
        _ = spec
        self._note("transpile_spec")
        if self.transpile_spec_plan:
            item = self.transpile_spec_plan.pop(0)
            if isinstance(item, UnsupportedConstruct):
                raise item
            return item
        return DAFNY_SPEC

    def fake_spec_cases(self, spec: Spec, function: FunctionRef, repo_root: Path, **kwargs: Any) -> Any:
        _ = (spec, function, repo_root, kwargs)
        self._note("spec_cases")
        return self.cases

    def fake_check_spec_translation(
        self, spec: Spec, dafny_spec: str, function: FunctionRef, repo_root: Path, **kwargs: Any
    ) -> CheckRecord:
        _ = (spec, dafny_spec, function, repo_root, kwargs)
        self._note("check_spec")
        if self.spec_check_record is not None:
            return self.spec_check_record
        return CheckRecord(CheckKind.DIFFERENTIAL, True, "")

    def fake_transpile(self, function: FunctionRef, **kwargs: Any) -> str:
        _ = kwargs
        self._note("transpile_function")
        if self.transpile_plan:
            item = self.transpile_plan.pop(0)
            if isinstance(item, UnsupportedConstruct):
                raise item
            return item
        return DAFNY_FUNCTION

    def fake_translate(self, function: FunctionRef, spec: Spec, **kwargs: Any) -> Translation:
        _ = spec
        self._note("translate", kwargs.get("model"))
        self.translate_count += 1
        mismatch = kwargs.get("mismatch")
        assert mismatch is None or isinstance(mismatch, Counterexample)
        self.translate_mismatches.append(mismatch)
        return Translation(
            function, TranslationPath.AI, TranslationPath.TRANSPILER, DAFNY_FUNCTION, kwargs["dafny_spec"]
        )

    def fake_equivalence(self, function: FunctionRef, translation: Translation, repo_root: Path, **kwargs: Any) -> Any:
        _ = (function, translation, repo_root)
        self._note("equivalence", kwargs.get("model"))
        if self.equivalence_plan:
            return self.equivalence_plan.pop(0)
        return EquivalenceReport(records=(CheckRecord(CheckKind.DIFFERENTIAL, True, "", 10, 10),), mismatch=None)

    def fake_prove(self, translation: Translation, assumptions: Sequence[Any], **kwargs: Any) -> ProofOutcome:
        _ = (translation, assumptions)
        self._note("prove")
        self.prove_lines.append(kwargs["function_lines"])
        with self._guard:
            self.finished[translation.function.qualname] = time.monotonic()
        if self.prove_raises is not None:
            raise self.prove_raises
        return self.prove_outcome

    def fake_search(self, function: FunctionRef, spec: Spec, repo_root: Path, **kwargs: Any) -> Counterexample | None:
        _ = (function, spec, repo_root)
        self._note("search", kwargs.get("model"))
        self.search_failed_clauses.append(kwargs.get("failed_clause"))
        return self.search_hit

    def fake_build(self, function: FunctionRef, spec: Spec, repo_root: Path, arguments: str) -> Counterexample:
        _ = (function, spec, repo_root)
        self._note("build")
        self.build_arguments.append(arguments)
        return Counterexample(arguments, "expected", "actual", "def test_x():\n    assert True\n")

    def fake_review(self, context: Context, spec: Spec, **kwargs: Any) -> SecondOpinion:
        _ = (context, spec)
        model = kwargs.get("model")
        assert isinstance(model, str)
        self._note("review", model)
        self.review_calls += 1
        self.review_models.append(model)
        return SecondOpinion(True, None, "agrees.")


def _install(monkeypatch: pytest.MonkeyPatch, steps: _Steps) -> None:
    monkeypatch.setattr(pcc.pipeline, "collect", steps.fake_collect)
    monkeypatch.setattr(pcc.pipeline, "search_further", steps.fake_search_further)
    monkeypatch.setattr(pcc.pipeline, "derive", steps.fake_derive)
    monkeypatch.setattr(pcc.pipeline, "write_spec", steps.fake_write)
    monkeypatch.setattr(pcc.pipeline, "check_spec", steps.fake_check)
    monkeypatch.setattr(pcc.pipeline, "transpile_spec", steps.fake_transpile_spec)
    monkeypatch.setattr(pcc.pipeline, "spec_cases", steps.fake_spec_cases)
    monkeypatch.setattr(pcc.pipeline, "check_spec_translation", steps.fake_check_spec_translation)
    monkeypatch.setattr(pcc.pipeline, "transpile_function", steps.fake_transpile)
    monkeypatch.setattr(pcc.pipeline, "translate", steps.fake_translate)
    monkeypatch.setattr(pcc.pipeline, "check_equivalence", steps.fake_equivalence)
    monkeypatch.setattr(pcc.pipeline, "prove", steps.fake_prove)
    monkeypatch.setattr(pcc.pipeline, "search_counterexample", steps.fake_search)
    monkeypatch.setattr(pcc.pipeline, "build_counterexample", steps.fake_build)
    monkeypatch.setattr(pcc.pipeline, "review", steps.fake_review)


def _run(
    ref: FunctionRef, tmp_path: Path, steps: _Steps, config: Config | None = None, answers: list[str] | None = None
) -> Result:
    return pcc.pipeline.run(
        ref, tmp_path, client=FakeClient(answers if answers is not None else []), config=config or Config()
    )


def _mismatch(arguments: str = "[1]") -> EquivalenceReport:
    return EquivalenceReport(
        records=(CheckRecord(CheckKind.DIFFERENTIAL, False, "differs", 10, 10),),
        mismatch=make_counterexample(arguments=arguments),
    )


# -- Example 1: the happy path --------------------------------------------


def test_happy_path_gives_proven(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert result.reason == ""
    assert result.intent is not None
    assert result.spec is not None
    assert result.translation is not None
    assert result.translation.checks, "the spec and equivalence records land on the translation"
    assert result.proof is not None and result.proof.proven
    assert result.usage == ()
    assert steps.prove_lines == [len(SIMPLE_SOURCE.splitlines())]
    assert steps.review_calls == 0, "a clean proof with no contradictions needs no second opinion"


def test_model_routing_uses_config_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.evidence = ()
    _install(monkeypatch, steps)
    config = Config(
        models=ModelsConfig(
            intent="m-intent",
            spec="m-spec",
            mutants="m-mutants",
            translation="m-translation",
            edge_cases="m-edge",
            counterexample="m-counter",
            second_opinion="m-second",
            context_search="m-search",
        )
    )

    result = _run(ref, tmp_path, steps, config)

    assert result.state is State.PROVEN
    assert steps.models["derive"] == ["m-intent"]
    assert steps.models["write"] == ["m-spec"]
    assert steps.models["check"] == ["m-mutants"]
    assert steps.models["equivalence"] == ["m-edge"]
    assert steps.models["search_further"] == ["m-search"]
    assert steps.translate_count == 0, "the transpiler path needs no AI translation"


def test_result_usage_is_the_wrapper_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.call_wrapped_client_in_derive = True
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps, answers=["{}"])

    assert result.state is State.PROVEN
    assert len(result.usage) == 1, "every Result carries the wrapper usage"


# -- Step 0: early not_checkable --------------------------------------------


def test_async_function_never_calls_a_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = "async def fetch(url: str) -> str:\n    return url\n"
    ref = function_ref(source=source, path=PATH, qualname="fetch", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.intent is None and result.spec is None and result.translation is None
    assert steps.calls == [], "no step runs for an async function"
    assert result.usage == ()


def test_generator_function_is_not_checkable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = "def gen(n: int) -> int:\n    yield n\n"
    ref = function_ref(source=source, path=PATH, qualname="gen", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert steps.calls == []


@pytest.mark.parametrize(
    "source", ["def f(x: int) -> int:\n    return eval(x)\n", "def f(x: int) -> int:\n    exec(x)\n"]
)
def test_eval_exec_is_not_checkable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str) -> None:
    ref = function_ref(source=source, path=PATH, qualname="f", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert steps.calls == []


@pytest.mark.parametrize(
    "source",
    [
        "import threading\ndef f(x: int) -> int:\n    return x\n",
        "from multiprocessing import Pool\ndef f(x: int) -> int:\n    return x\n",
    ],
)
def test_threading_is_not_checkable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str) -> None:
    ref = function_ref(source=source, path=PATH, qualname="f", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert steps.calls == []


def test_method_is_not_checkable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = "class Cart:\n    def total(self, xs: list[int]) -> int:\n        return sum(xs)\n"
    (tmp_path / "cart.py").write_text(module, encoding="utf-8")
    ref = function_ref(
        source="    def total(self, xs: list[int]) -> int:\n        return sum(xs)\n",
        path="cart.py",
        qualname="Cart.total",
        lineno=2,
    )
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "methods are tier 3"
    assert steps.calls == []


def test_staticmethod_is_checked_normally(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = "class Cart:\n    @staticmethod\n    def total(xs: list[int]) -> int:\n        return sum(xs)\n"
    (tmp_path / "cart.py").write_text(module, encoding="utf-8")
    ref = function_ref(
        source="    @staticmethod\n    def total(xs: list[int]) -> int:\n        return sum(xs)\n",
        path="cart.py",
        qualname="Cart.total",
        lineno=3,
    )
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert "collect" in steps.calls


def test_missing_type_hints_end_before_any_model_call(tmp_path: Path) -> None:
    ref = function_ref(source="def f(x):\n    return x\n", path=PATH, qualname="f", lineno=1)
    client = FakeClient([])

    result = pcc.pipeline.run(ref, tmp_path, client=client, config=Config())

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "no types"
    assert client.requests == [], "no model call happens before the strategies check"


def test_self_recursion_reaches_step_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = (
        "def is_palindrome_recursive(s: str) -> bool:\n"
        "    if len(s) <= 1:\n"
        "        return True\n"
        "    return s[0] == s[-1] and is_palindrome_recursive(s[1:-1])\n"
    )
    ref = function_ref(source=source, path=PATH, qualname="is_palindrome_recursive", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert steps.calls[0] == "collect"


# -- Steps 1 and 2 -----------------------------------------------------------


def test_search_agent_runs_only_without_strong_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.evidence = ()
    _install(monkeypatch, steps)

    _run(ref, tmp_path, steps)

    assert "search_further" in steps.calls


def test_search_agent_skipped_with_docstring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.evidence = (Evidence(SourceKind.DOCSTRING, "doc", "the total"),)
    _install(monkeypatch, steps)

    _run(ref, tmp_path, steps)

    assert "search_further" not in steps.calls


def test_search_agent_disabled_by_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.evidence = ()
    _install(monkeypatch, steps)
    config = Config(pipeline=PipelineConfig(search_agent=False))

    _run(ref, tmp_path, steps, config)

    assert "search_further" not in steps.calls


def test_intent_not_checkable_ends_not_checkable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.intent_raises = NotCheckable("the sources contradict each other")
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "the sources contradict each other"
    assert "write" not in steps.calls
    assert steps.review_calls == 0, "no context/spec pair exists to review"


# -- Step 3: spec check, rewrites, counterexample ------------------------------


def test_mutant_failure_rewrites_with_detail_then_proves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.check_plan = [
        [
            CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 10, 10),
            CheckRecord(CheckKind.MUTANTS, False, "survivor"),
        ],
        _passing_checks(),
    ]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert steps.write_hints == ["", "survivor"]
    assert steps.calls.count("check") == 2


def test_mutant_exhaustion_is_not_checkable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.check_plan = [
        [CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 10, 10), CheckRecord(CheckKind.MUTANTS, False, "s1")],
        [CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 10, 10), CheckRecord(CheckKind.MUTANTS, False, "s2")],
        [CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 10, 10), CheckRecord(CheckKind.MUTANTS, False, "s3")],
    ]
    _install(monkeypatch, steps)
    config = Config(spec=SpecConfig(max_rewrites=2))

    result = _run(ref, tmp_path, steps, config)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "spec does not kill mutants"
    assert steps.write_hints == ["", "s1", "s2"]
    assert steps.review_calls == 0


def test_hypothesis_failure_builds_counterexample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.check_plan = [_hyp_fail("[-1]")]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.COUNTEREXAMPLE
    assert result.counterexample is not None
    assert result.counterexample.arguments == "[-1]"
    assert "transpile_spec" not in steps.calls, "translation never runs after a step-3 counterexample"
    assert "prove" not in steps.calls
    assert steps.review_calls == 1


def test_hypothesis_arguments_strip_raises_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.check_plan = [_hyp_fail("[-1] raises ValueError")]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.COUNTEREXAMPLE
    assert steps.build_arguments == ["[-1]"]
    assert result.counterexample is not None
    assert result.counterexample.arguments == "[-1]"


def test_candidate_fix_becomes_suggested_fix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.check_plan = [
        [
            CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, "[-1]", 10, 10),
            CheckRecord(CheckKind.MUTANTS, True, ""),
            CheckRecord(CheckKind.CANDIDATE_FIX, True, "--- a\n+++ b\n"),
        ]
    ]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.COUNTEREXAMPLE
    assert result.counterexample is not None
    assert result.counterexample.suggested_fix == "--- a\n+++ b\n"


def test_precondition_never_holds_overrides_rewrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.check_plan = [
        [
            CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, "precondition never holds", 0, 10),
            CheckRecord(CheckKind.MUTANTS, False, "precondition never holds"),
        ]
    ]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "precondition never holds"
    assert steps.write_hints == [""], "no rewrite happens for a vacuous precondition"


def test_original_timeout_is_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.check_plan = [
        [
            CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, "timeout", 5, 10),
            CheckRecord(CheckKind.MUTANTS, True, ""),
        ]
    ]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.OPEN
    assert result.reason == "original times out"
    assert "build" not in steps.calls
    assert steps.review_calls == 1, "an open result gets the second opinion"


# -- Step 4: translation -------------------------------------------------------


def test_transpile_spec_unsupported_rewrites_then_rechecks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_spec_plan = [UnsupportedConstruct("call foo"), DAFNY_SPEC]
    steps.check_plan = [_passing_checks(), _passing_checks()]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert steps.write_hints == ["", "the spec transpiler does not support: call foo"]
    assert steps.calls.count("check") == 2, "the rewritten spec runs the full step-3 check again"


def test_transpile_spec_rewrite_reaches_counterexample_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_spec_plan = [UnsupportedConstruct("call foo"), DAFNY_SPEC]
    steps.check_plan = [_passing_checks(), _hyp_fail("[-1] raises ValueError")]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.COUNTEREXAMPLE
    assert steps.build_arguments == ["[-1]"]
    assert "prove" not in steps.calls
    assert steps.review_calls == 1


def test_transpile_spec_exhaustion_is_not_translatable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_spec_plan = [UnsupportedConstruct("call foo")]
    _install(monkeypatch, steps)
    config = Config(spec=SpecConfig(max_rewrites=0))

    result = _run(ref, tmp_path, steps, config)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "spec not translatable"


def test_shared_rewrite_budget_between_mutants_and_transpiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.check_plan = [
        [CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, True, "", 10, 10), CheckRecord(CheckKind.MUTANTS, False, "s1")],
        _passing_checks(),
    ]
    steps.transpile_spec_plan = [UnsupportedConstruct("call foo")]
    _install(monkeypatch, steps)
    config = Config(spec=SpecConfig(max_rewrites=1))

    result = _run(ref, tmp_path, steps, config)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "spec not translatable", "the MUTANTS rewrite used the shared counter"
    assert steps.write_hints == ["", "s1"]


def test_transpile_spec_no_types_has_no_rewrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_spec_plan = [UnsupportedConstruct("no types")]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "no types"
    assert steps.write_hints == [""]
    assert "transpile_function" not in steps.calls
    assert "translate" not in steps.calls


@pytest.mark.parametrize("construct", ["type float", "type Optional[int]", "type int | None"])
def test_transpile_spec_unsupported_type_has_no_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, construct: str
) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_spec_plan = [UnsupportedConstruct(construct)]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == construct
    assert steps.write_hints == [""]
    assert "translate" not in steps.calls


def test_transpile_spec_tuple_type_rewrites(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_spec_plan = [UnsupportedConstruct("type tuple"), DAFNY_SPEC]
    steps.check_plan = [_passing_checks(), _passing_checks()]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert steps.write_hints[1] == "the spec transpiler does not support: type tuple"


def test_transpile_function_unsupported_type_has_no_ai_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_plan = [UnsupportedConstruct("type float")]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "type float"
    assert steps.translate_count == 0


def test_transpile_function_no_types_has_no_ai_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_plan = [UnsupportedConstruct("no types")]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "no types"
    assert steps.translate_count == 0


def test_transpile_function_tuple_type_uses_ai_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_plan = [UnsupportedConstruct("type tuple")]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert result.translation is not None
    assert result.translation.function_path is TranslationPath.AI


def test_spec_translation_difference_is_not_checkable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.spec_check_record = CheckRecord(CheckKind.DIFFERENTIAL, False, "case 3 differs")
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "spec translation differs"
    assert "transpile_function" not in steps.calls
    assert "prove" not in steps.calls


def test_transpiler_mismatch_falls_back_to_ai(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.equivalence_plan = [
        _mismatch("[1]"),
        EquivalenceReport(records=(CheckRecord(CheckKind.DIFFERENTIAL, True, "", 10, 10),), mismatch=None),
    ]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert result.translation is not None
    assert result.translation.function_path is TranslationPath.AI
    details = [record.detail for record in result.translation.checks]
    assert "transpiler mismatch" in details
    assert steps.translate_mismatches == [make_counterexample(arguments="[1]")]


def test_ai_translation_mismatch_exhaustion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.transpile_plan = [UnsupportedConstruct("comprehension")]
    steps.equivalence_plan = [_mismatch(), _mismatch(), _mismatch(), _mismatch(), _mismatch()]
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "translation not equivalent"
    assert steps.translate_count == 4, "one AI translation plus three retranslations"


# -- Steps 5 and 6 -------------------------------------------------------------


def test_failed_proof_searches_with_failed_clause(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.prove_outcome = ProofOutcome(
        False,
        "source",
        2,
        "spark",
        "assertion might not hold",
        (DafnyError(3, 1, "assertion might not hold", "spec(xs, result)"),),
    )
    steps.search_hit = make_counterexample()
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.COUNTEREXAMPLE
    assert result.counterexample == make_counterexample()
    assert steps.search_failed_clauses == ["spec(xs, result)"]
    assert steps.review_calls == 1


def test_failed_proof_without_related_clause_searches_with_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.prove_outcome = ProofOutcome(False, "source", 2, "spark", "assertion might not hold", ())
    steps.search_hit = None
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.OPEN
    assert result.reason == "no proof found", "step 6 fixes the reason even with a prover last error"
    assert steps.search_failed_clauses == [None]
    assert steps.review_calls == 1


def test_no_counterexample_after_failed_proof_is_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.prove_outcome = ProofOutcome(False, "source", 2, "spark", "no proof found", ())
    steps.search_hit = None
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.OPEN
    assert result.reason == "no proof found"


# -- Step 7: second opinion ------------------------------------------------------


def test_second_opinion_is_advisory_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.prove_outcome = ProofOutcome(False, "source", 2, "spark", "no proof found", ())
    steps.search_hit = make_counterexample()
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.COUNTEREXAMPLE
    assert result.counterexample == make_counterexample(), "the review never changes the counterexample"
    assert result.second_opinion is not None
    assert steps.review_calls == 1


def test_contradictions_trigger_review_on_proven(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    intent = make_intent(ref)
    steps.intent_value = dataclasses.replace(intent, contradictions=(make_contradiction(),))
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.PROVEN
    assert result.second_opinion is not None
    assert steps.review_calls == 1


def test_second_opinion_uses_configured_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.prove_outcome = ProofOutcome(False, "source", 1, "spark", "no proof found", ())
    steps.search_hit = None
    _install(monkeypatch, steps)
    config = Config(models=ModelsConfig(second_opinion="m-second"))

    result = _run(ref, tmp_path, steps, config)

    assert result.state is State.OPEN
    assert steps.review_models == ["m-second"]


# -- Budget and errors -----------------------------------------------------------


def test_budget_exceeded_in_prove_is_open_with_partials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.prove_raises = BudgetExceeded("budget exceeded")
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.OPEN
    assert result.reason == "budget"
    assert result.intent is not None
    assert result.spec is not None
    assert result.translation is not None
    assert "search" not in steps.calls
    assert steps.review_calls == 0


def test_elapsed_budget_stops_before_any_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)
    config = Config(budget=BudgetConfig(max_seconds_per_function=0.0))

    result = _run(ref, tmp_path, steps, config)

    assert result.state is State.OPEN
    assert result.reason == "budget"
    assert steps.calls == []
    assert result.intent is None


def test_tool_error_escapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.collect_raises = ToolError("disk is gone")
    _install(monkeypatch, steps)

    with pytest.raises(ToolError):
        _run(ref, tmp_path, steps)


@pytest.mark.parametrize(
    "step",
    [
        "collect",
        "derive",
        "write_spec",
        "check_spec",
        "transpile_spec",
        "spec_cases",
        "check_spec_translation",
        "transpile_function",
        "translate",
        "check_equivalence",
        "prove",
        "search_counterexample",
        "build_counterexample",
        "review",
    ],
)
def test_tool_error_escapes_from_each_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, step: str) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    if step == "build_counterexample":
        steps.check_plan = [_hyp_fail("[-1]")]
    if step == "translate":
        steps.transpile_plan = [UnsupportedConstruct("comprehension")]
    if step in ("search_counterexample", "review"):
        steps.prove_outcome = ProofOutcome(False, "source", 1, "spark", "no proof found", ())
        steps.search_hit = None
    _install(monkeypatch, steps)

    def boom(*args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        raise ToolError("boom")

    monkeypatch.setattr(pcc.pipeline, step, boom)

    with pytest.raises(ToolError):
        _run(ref, tmp_path, steps)


# -- run_many --------------------------------------------------------------------


def _order_refs() -> tuple[FunctionRef, FunctionRef, FunctionRef]:
    a_source = "def a(x: int) -> int:\n    return c(x)\n"
    b_source = "def b(x: int) -> int:\n    return c(x) + 1\n"
    c_source = "def c(x: int) -> int:\n    return x\n"
    a = function_ref(source=a_source, path=PATH, qualname="a", lineno=1)
    b = function_ref(source=b_source, path=PATH, qualname="b", lineno=4)
    c = function_ref(source=c_source, path=PATH, qualname="c", lineno=7)
    return a, b, c


def test_run_many_keeps_order_and_levels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    a, b, c = _order_refs()
    steps = _Steps(a)

    def slow_prove(translation: Translation, assumptions: Sequence[Any], **kwargs: Any) -> ProofOutcome:
        outcome = steps.fake_prove(translation, assumptions, **kwargs)
        if translation.function.qualname == "c":
            time.sleep(0.05)
        return outcome

    monkeypatch.setattr(pcc.pipeline, "collect", steps.fake_collect)
    monkeypatch.setattr(pcc.pipeline, "search_further", steps.fake_search_further)
    monkeypatch.setattr(pcc.pipeline, "derive", steps.fake_derive)
    monkeypatch.setattr(pcc.pipeline, "write_spec", steps.fake_write)
    monkeypatch.setattr(pcc.pipeline, "check_spec", steps.fake_check)
    monkeypatch.setattr(pcc.pipeline, "transpile_spec", steps.fake_transpile_spec)
    monkeypatch.setattr(pcc.pipeline, "spec_cases", steps.fake_spec_cases)
    monkeypatch.setattr(pcc.pipeline, "check_spec_translation", steps.fake_check_spec_translation)
    monkeypatch.setattr(pcc.pipeline, "transpile_function", steps.fake_transpile)
    monkeypatch.setattr(pcc.pipeline, "translate", steps.fake_translate)
    monkeypatch.setattr(pcc.pipeline, "check_equivalence", steps.fake_equivalence)
    monkeypatch.setattr(pcc.pipeline, "prove", slow_prove)
    monkeypatch.setattr(pcc.pipeline, "search_counterexample", steps.fake_search)
    monkeypatch.setattr(pcc.pipeline, "build_counterexample", steps.fake_build)
    monkeypatch.setattr(pcc.pipeline, "review", steps.fake_review)

    results = pcc.pipeline.run_many([a, b, c], tmp_path, client=FakeClient([]), config=Config())

    assert [result.state for result in results] == [State.PROVEN, State.PROVEN, State.PROVEN]
    assert [result.function.qualname for result in results] == ["a", "b", "c"]
    assert steps.finished["c"] < steps.started["a"], "level 2 starts after level 1"
    assert steps.finished["c"] < steps.started["b"]


def test_run_many_budget_is_per_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    a, _, _ = _order_refs()
    independents = [a, function_ref(source="def z(x: int) -> int:\n    return x\n", path=PATH, qualname="z", lineno=10)]
    steps = _Steps(a)

    def selective_prove(translation: Translation, assumptions: Sequence[Any], **kwargs: Any) -> ProofOutcome:
        if translation.function.qualname == "a":
            raise BudgetExceeded("budget exceeded")
        return steps.fake_prove(translation, assumptions, **kwargs)

    monkeypatch.setattr(pcc.pipeline, "collect", steps.fake_collect)
    monkeypatch.setattr(pcc.pipeline, "search_further", steps.fake_search_further)
    monkeypatch.setattr(pcc.pipeline, "derive", steps.fake_derive)
    monkeypatch.setattr(pcc.pipeline, "write_spec", steps.fake_write)
    monkeypatch.setattr(pcc.pipeline, "check_spec", steps.fake_check)
    monkeypatch.setattr(pcc.pipeline, "transpile_spec", steps.fake_transpile_spec)
    monkeypatch.setattr(pcc.pipeline, "spec_cases", steps.fake_spec_cases)
    monkeypatch.setattr(pcc.pipeline, "check_spec_translation", steps.fake_check_spec_translation)
    monkeypatch.setattr(pcc.pipeline, "transpile_function", steps.fake_transpile)
    monkeypatch.setattr(pcc.pipeline, "translate", steps.fake_translate)
    monkeypatch.setattr(pcc.pipeline, "check_equivalence", steps.fake_equivalence)
    monkeypatch.setattr(pcc.pipeline, "prove", selective_prove)
    monkeypatch.setattr(pcc.pipeline, "search_counterexample", steps.fake_search)
    monkeypatch.setattr(pcc.pipeline, "build_counterexample", steps.fake_build)
    monkeypatch.setattr(pcc.pipeline, "review", steps.fake_review)

    results = pcc.pipeline.run_many(independents, tmp_path, client=FakeClient([]), config=Config())

    assert [result.state for result in results] == [State.OPEN, State.PROVEN]
    assert results[0].reason == "budget"
    assert results[1].proof is not None


def test_run_many_tool_error_fails_the_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    a, _, _ = _order_refs()
    z = function_ref(source="def z(x: int) -> int:\n    return x\n", path=PATH, qualname="z", lineno=10)
    steps = _Steps(a)
    steps.collect_raises = ToolError("boom")
    _install(monkeypatch, steps)

    with pytest.raises(ToolError):
        pcc.pipeline.run_many([a, z], tmp_path, client=FakeClient([]), config=Config())


def test_equivalence_uses_edge_case_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)
    config = Config(models=ModelsConfig(edge_cases="m-edge"))

    result = _run(ref, tmp_path, steps, config)

    assert result.state is State.PROVEN
    assert steps.models["equivalence"] == ["m-edge"]


def test_search_uses_counterexample_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.prove_outcome = ProofOutcome(False, "source", 1, "spark", "no proof found", ())
    _install(monkeypatch, steps)
    config = Config(models=ModelsConfig(counterexample="m-counter"))

    _run(ref, tmp_path, steps, config)

    assert steps.models["search"] == ["m-counter"]


def test_check_uses_mutants_model_and_prover_stages_pass_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    seen: dict[str, Any] = {}
    real_prove = steps.fake_prove

    def recording_prove(translation: Translation, assumptions: Sequence[Any], **kwargs: Any) -> ProofOutcome:
        seen["stages"] = kwargs["stages"]
        return real_prove(translation, assumptions, **kwargs)

    _install(monkeypatch, steps)
    monkeypatch.setattr(pcc.pipeline, "prove", recording_prove)
    config = Config(models=ModelsConfig(mutants="m-mutants"))

    _run(ref, tmp_path, steps, config)

    assert steps.models["check"] == ["m-mutants"]
    assert seen["stages"] == config.proof.prover_stages


def test_proven_translation_carries_all_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.translation is not None
    kinds = [record.kind for record in result.translation.checks]
    assert CheckKind.DIFFERENTIAL in kinds
    assert all(record.passed for record in result.translation.checks)


def test_counterexample_model_used_for_nothing_else(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ref = function_ref(source=SIMPLE_SOURCE, path=PATH, qualname="total", lineno=1)
    steps = _Steps(ref)
    steps.prove_outcome = ProofOutcome(False, "source", 1, "spark", "no proof found", ())
    steps.search_hit = make_counterexample()
    _install(monkeypatch, steps)

    result = _run(ref, tmp_path, steps)

    assert result.state is State.COUNTEREXAMPLE
    assert "search" in steps.models
    assert steps.models["search"] == [Config().models.counterexample]
