"""Pipeline calls inside the project (issue #43).

Callees are proven first and callers use only the callee's spec: a proven callee passes its certificate source as an
extra module plus an import, an unproven callee becomes an assumption, mutual recursion and counterexample callees
end the caller ``open``. Every test monkeypatches the steps and asserts the callee info on the mocks.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import pcc.pipeline
from pcc.certificate import from_result
from pcc.config import Config
from pcc.core import (
    Assumption,
    AssumptionKind,
    CheckKind,
    CheckRecord,
    Context,
    Counterexample,
    DafnyError,
    Dependency,
    EquivalenceReport,
    FunctionRef,
    Intent,
    ProofOutcome,
    Result,
    SecondOpinion,
    Spec,
    State,
    Translation,
    TranslationPath,
    UnsupportedConstruct,
    Usage,
)
from pcc.models import FakeClient
from pcc.verifier.layout import module_name
from tests.helpers import counterexample as make_counterexample
from tests.helpers import function_ref
from tests.helpers import intent as make_intent
from tests.helpers import proof as make_proof
from tests.helpers import result as make_result
from tests.helpers import spec as make_spec
from tests.helpers import translation as make_translation

PATH = "pkg/calls.py"

G_SOURCE = "def g(x: int) -> int:\n    return x\n"
F_SOURCE = "def f(x: int) -> int:\n    return g(x)\n"
FA_SOURCE = "def fa(x: int) -> int:\n    return a(x)\n"
A_SOURCE = "def a(x: int) -> int:\n    return b(x)\n"
B_SOURCE = "def b(x: int) -> int:\n    return a(x)\n"
H_SOURCE = "def h(x: int) -> int:\n    return x\n"

G_DAFNY_SPEC = "predicate spec(x: int, result: int)\n{\n  result == x\n}\n"
G_DAFNY_FUNCTION = (
    "method g(x: int) returns (result: int)\n  requires pre(x)\n  ensures spec(x, result)\n{\n  result := x;\n}\n"
)
G_PROOF_SOURCE = "module PCC_pkg_calls__g_deadbeef {\n  import opened PyLib\n}\n"

CALLER_DAFNY_SPEC = "predicate spec(x: int, result: int)\n{\n  result == g(x)\n}\n"
CALLER_DAFNY_FUNCTION = (
    "method f(x: int) returns (result: int)\n  requires pre(x)\n  ensures spec(x, result)\n{\n  result := g(x);\n}\n"
)
CALLER_PROOF_SOURCE = "module PCC_pkg_calls__f_deadbeef {\n  import opened PyLib\n}\n"


def g_ref() -> FunctionRef:
    return function_ref(source=G_SOURCE, path=PATH, qualname="g", lineno=1)


def f_ref() -> FunctionRef:
    return function_ref(source=F_SOURCE, path=PATH, qualname="f", lineno=5)


def fa_ref() -> FunctionRef:
    return function_ref(source=FA_SOURCE, path=PATH, qualname="fa", lineno=9)


def a_ref() -> FunctionRef:
    return function_ref(source=A_SOURCE, path=PATH, qualname="a", lineno=13)


def b_ref() -> FunctionRef:
    return function_ref(source=B_SOURCE, path=PATH, qualname="b", lineno=17)


def proven_callee(ref: FunctionRef) -> Result:
    """A proven callee result carrying its spec and its certificate source."""
    return Result(
        ref,
        State.PROVEN,
        "",
        make_intent(ref),
        make_spec(ref),
        make_translation(ref, dafny_function=G_DAFNY_FUNCTION, dafny_spec=G_DAFNY_SPEC),
        (),
        None,
        make_proof(dafny_source=G_PROOF_SOURCE),
    )


def open_callee(ref: FunctionRef) -> Result:
    """An unproven callee result carrying its translated method and spec."""
    return Result(
        ref,
        State.OPEN,
        "no proof found",
        make_intent(ref),
        make_spec(ref),
        make_translation(ref, dafny_function=G_DAFNY_FUNCTION, dafny_spec=G_DAFNY_SPEC),
        (),
        None,
        None,
    )


class _Recorded:
    """What the monkeypatched steps received; the tests assert the callee info on these."""

    def __init__(self) -> None:
        self.transpile_callees: list[dict[str, str]] = []
        self.prove_extra: list[tuple[str, ...]] = []
        self.prove_imports: list[tuple[str, ...]] = []
        self.prove_assumptions: list[tuple[Assumption, ...]] = []
        self.prove_assumptions_module: list[str] = []
        self.prove_outcome = ProofOutcome(True, CALLER_PROOF_SOURCE, 1, "glimmer")
        self.counterexample: Counterexample | None = None
        self.transpile_fails = False


def _install_steps(monkeypatch: pytest.MonkeyPatch, recorded: _Recorded) -> None:
    """Replace every step ``run`` calls with a recording fake."""

    def fake_collect(function: FunctionRef, repo_root: Path) -> Context:
        _ = repo_root
        return Context(function, ())

    def fake_search_further(
        context: Context, repo_root: Path, *, client: Any, model: str, max_turns: int = 8
    ) -> Context:
        _ = (repo_root, client, model, max_turns)
        return context

    def fake_derive(context: Context, *, client: Any, model: str) -> Intent:
        _ = (client, model)
        return make_intent(context.function)

    def fake_write(intent: Intent, *, client: Any, model: str, hint: str = "") -> Spec:
        _ = (client, model, hint)
        return make_spec(intent.function)

    def fake_check(spec: Spec, function: FunctionRef, repo_root: Path, *, client: Any, model: str, config: Any) -> Spec:
        _ = (function, repo_root, client, model, config)
        return spec

    def fake_transpile_spec(spec: Spec) -> str:
        _ = spec
        return CALLER_DAFNY_SPEC

    def fake_transpile(function: FunctionRef, *, callees: Mapping[str, str] = MappingProxyType({})) -> str:
        _ = function
        recorded.transpile_callees.append(dict(callees))
        if recorded.transpile_fails:
            raise UnsupportedConstruct("project call")
        return CALLER_DAFNY_FUNCTION

    def fake_translate(
        function: FunctionRef,
        spec: Spec,
        *,
        client: Any,
        model: str,
        dafny_spec: str,
        mismatch: Counterexample | None = None,
        resolve_retries: int = 2,
    ) -> Translation:
        _ = (spec, client, model, mismatch, resolve_retries)
        return Translation(function, TranslationPath.AI, TranslationPath.TRANSPILER, CALLER_DAFNY_FUNCTION, dafny_spec)

    def fake_equivalence(
        function: FunctionRef,
        translation: Translation,
        repo_root: Path,
        *,
        inputs: int,
        symbolic_seconds: float,
        client: Any,
        model: str,
    ) -> EquivalenceReport:
        _ = (function, translation, repo_root, inputs, symbolic_seconds, client, model)
        return EquivalenceReport(
            records=(CheckRecord(CheckKind.DIFFERENTIAL, True, "", 10, 10),),
            mismatch=None,
        )

    def fake_prove(
        translation: Translation,
        assumptions: Sequence[Assumption],
        *,
        client: Any,
        stages: Any,
        resource_limit: int,
        function_lines: int,
        extra_modules: Sequence[str] = (),
        assumptions_module: str = "",
        imports: Sequence[str] = (),
    ) -> ProofOutcome:
        _ = (translation, client, stages, resource_limit, function_lines)
        recorded.prove_extra.append(tuple(extra_modules))
        recorded.prove_imports.append(tuple(imports))
        recorded.prove_assumptions.append(tuple(assumptions))
        recorded.prove_assumptions_module.append(assumptions_module)
        return recorded.prove_outcome

    def fake_search(
        function: FunctionRef,
        spec: Spec,
        repo_root: Path,
        *,
        failed_clause: str | None,
        client: Any,
        model: str,
        config: Any,
    ) -> Counterexample | None:
        _ = (function, spec, repo_root, failed_clause, client, model, config)
        return recorded.counterexample

    def fake_review(context: Context, spec: Spec, *, client: Any, model: str) -> SecondOpinion:
        _ = (context, spec, client, model)
        return SecondOpinion(True, None, "agrees.")

    monkeypatch.setattr(pcc.pipeline, "collect", fake_collect)
    monkeypatch.setattr(pcc.pipeline, "search_further", fake_search_further)
    monkeypatch.setattr(pcc.pipeline, "derive", fake_derive)
    monkeypatch.setattr(pcc.pipeline, "write_spec", fake_write)
    monkeypatch.setattr(pcc.pipeline, "check_spec", fake_check)
    monkeypatch.setattr(pcc.pipeline, "transpile_spec", fake_transpile_spec)
    monkeypatch.setattr(pcc.pipeline, "transpile_function", fake_transpile)
    monkeypatch.setattr(pcc.pipeline, "translate", fake_translate)
    monkeypatch.setattr(pcc.pipeline, "check_equivalence", fake_equivalence)
    monkeypatch.setattr(pcc.pipeline, "prove", fake_prove)
    monkeypatch.setattr(pcc.pipeline, "search_counterexample", fake_search)
    monkeypatch.setattr(pcc.pipeline, "review", fake_review)


def _run(function: FunctionRef, tmp_path: Path, callees: Mapping[FunctionRef, Result] | None = None) -> Result:
    return pcc.pipeline.run(
        function,
        tmp_path,
        client=FakeClient([]),
        config=Config(),
        callees=callees if callees is not None else {},
    )


def test_proven_callee_passes_module_and_import_with_no_assumptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()

    result = _run(f_ref(), tmp_path, {callee: proven_callee(callee)})

    assert result.state is State.PROVEN
    assert result.assumptions == ()
    assert recorded.transpile_callees == [{"g": module_name(callee)}]
    assert recorded.prove_extra == [(G_PROOF_SOURCE,)]
    assert recorded.prove_imports == [(module_name(callee),)]
    assert recorded.prove_assumptions_module == [""]


def test_open_callee_becomes_unproven_callee_assumption(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()

    result = _run(f_ref(), tmp_path, {callee: open_callee(callee)})

    assert result.state is State.PROVEN
    assert len(result.assumptions) == 1
    assumption = result.assumptions[0]
    assert assumption.kind is AssumptionKind.UNPROVEN_CALLEE
    assert assumption.subject == "g"
    assert "method {:axiom} g(" in assumption.dafny
    assert recorded.prove_extra == [()]
    (module,) = recorded.prove_assumptions_module
    match = re.fullmatch(r"module (PCC_assume_[0-9a-f]{8}) \{\n(?:.*\n)*\}\n", module)
    assert match is not None
    (assume_name,) = match.groups()
    assert recorded.prove_imports == [(assume_name,)]
    assert module.startswith(f"module {assume_name} {{\n")
    assert "predicate spec(x: int, result: int)" in module
    assert "result == x" in module
    assert "method {:axiom} g(x: int) returns (result: int)\n    requires pre(x)\n    ensures spec(x, result)" in module


SET_DISPLAY_FUNCTION = (
    "method g(x: int) returns (result: int)\n  requires pre(x)\n  ensures result in {0, 1}\n{\n  result := x;\n}\n"
)


def test_open_callee_with_braces_in_contract_keeps_full_header(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()
    translation = make_translation(callee, dafny_function=SET_DISPLAY_FUNCTION, dafny_spec=G_DAFNY_SPEC)
    outcome = Result(
        callee,
        State.OPEN,
        "no proof found",
        make_intent(callee),
        make_spec(callee),
        translation,
        (),
        None,
        None,
    )

    result = _run(f_ref(), tmp_path, {callee: outcome})

    assert result.state is State.PROVEN
    (assumption,) = result.assumptions
    assert assumption.dafny.count("{:axiom}") == 1
    assert "ensures result in {0, 1}" in assumption.dafny
    (module,) = recorded.prove_assumptions_module
    assert "ensures result in {0, 1}" in module


BRACE_STRING_FUNCTION = (
    "method g(s: string) returns (result: bool)\n"
    '  requires s != "{"\n'
    "  ensures spec(s, result)\n"
    "{\n"
    '  result := s != "{";\n'
    "}\n"
)


def test_open_callee_with_brace_in_string_keeps_full_header(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()
    translation = make_translation(callee, dafny_function=BRACE_STRING_FUNCTION, dafny_spec=G_DAFNY_SPEC)
    outcome = Result(
        callee,
        State.OPEN,
        "no proof found",
        make_intent(callee),
        make_spec(callee),
        translation,
        (),
        None,
        None,
    )

    result = _run(f_ref(), tmp_path, {callee: outcome})

    assert result.state is State.PROVEN
    (assumption,) = result.assumptions
    assert assumption.dafny.count("{:axiom}") == 1
    assert 'requires s != "{"' in assumption.dafny
    (module,) = recorded.prove_assumptions_module
    assert 'requires s != "{"' in module


COMMENT_BRACE_FUNCTION = (
    "method g(x: int) returns (result: int)\n"
    "  requires pre(x) // { checked later\n"
    "  ensures spec(x, result)\n"
    "{\n"
    "  result := x;\n"
    "}\n"
)


def test_open_callee_with_brace_in_comment_keeps_full_header(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()
    translation = make_translation(callee, dafny_function=COMMENT_BRACE_FUNCTION, dafny_spec=G_DAFNY_SPEC)
    outcome = Result(
        callee,
        State.OPEN,
        "no proof found",
        make_intent(callee),
        make_spec(callee),
        translation,
        (),
        None,
        None,
    )

    result = _run(f_ref(), tmp_path, {callee: outcome})

    assert result.state is State.PROVEN
    (assumption,) = result.assumptions
    assert assumption.dafny.count("{:axiom}") == 1
    assert "requires pre(x) // { checked later" in assumption.dafny
    (module,) = recorded.prove_assumptions_module
    assert "requires pre(x) // { checked later" in module


def test_mutual_recursion_ends_open_without_prove(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("steps must not run for mutual recursion")

    monkeypatch.setattr(pcc.pipeline, "collect", fail)
    monkeypatch.setattr(pcc.pipeline, "prove", fail)

    results = pcc.pipeline.run_many([a_ref(), b_ref()], tmp_path, client=FakeClient([]), config=Config())

    assert [result.state for result in results] == [State.OPEN, State.OPEN]
    assert [result.reason for result in results] == ["mutual recursion", "mutual recursion"]


def test_counterexample_callee_opens_caller_without_prove(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("steps must not run after a counterexample callee")

    monkeypatch.setattr(pcc.pipeline, "collect", fail)
    monkeypatch.setattr(pcc.pipeline, "prove", fail)
    callee = g_ref()

    result = _run(f_ref(), tmp_path, {callee: make_result(callee, State.COUNTEREXAMPLE)})

    assert result.state is State.OPEN
    assert result.reason == "callee g has a counterexample"


def test_not_checkable_callee_makes_caller_not_checkable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("steps must not run after a not-checkable callee")

    monkeypatch.setattr(pcc.pipeline, "collect", fail)
    monkeypatch.setattr(pcc.pipeline, "prove", fail)
    callee = g_ref()

    result = _run(f_ref(), tmp_path, {callee: make_result(callee, State.NOT_CHECKABLE)})

    assert result.state is State.NOT_CHECKABLE
    assert result.reason == "callee g not checkable"


def test_mutual_recursion_reason_propagates_to_caller(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("prove must not run for a mutual recursion callee")

    monkeypatch.setattr(pcc.pipeline, "prove", fail)
    callee = a_ref()

    result = _run(fa_ref(), tmp_path, {callee: make_result(callee, State.OPEN, reason="mutual recursion")})

    assert result.state is State.OPEN
    assert result.reason == "mutual recursion"


def test_self_call_uses_no_callee_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    function = function_ref(source=H_SOURCE, path=PATH, qualname="h", lineno=21)

    result = _run(function, tmp_path, {function: proven_callee(function)})

    assert result.state is State.PROVEN
    assert result.assumptions == ()
    assert recorded.transpile_callees == [{}]
    assert recorded.prove_extra == [()]
    assert recorded.prove_imports == [()]
    assert recorded.prove_assumptions_module == [""]


def test_run_many_passes_earlier_level_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()
    caller = f_ref()

    results = pcc.pipeline.run_many([caller, callee], tmp_path, client=FakeClient([]), config=Config())

    assert [result.state for result in results] == [State.PROVEN, State.PROVEN]
    assert recorded.transpile_callees == [{}, {"g": module_name(callee)}]
    assert results[1].proof is not None
    assert recorded.prove_extra == [(), (results[1].proof.dafny_source,)]
    assert recorded.prove_imports == [(), (module_name(callee),)]
    assert recorded.prove_assumptions_module == ["", ""]


def test_dependency_hashes_use_callee_specs(tmp_path: Path) -> None:
    _ = tmp_path
    callee = g_ref()
    caller = f_ref()
    expected = hashlib.sha256(G_DAFNY_SPEC.encode("utf-8")).hexdigest()

    assert pcc.pipeline.dependency_hashes(caller, {callee: proven_callee(callee)}) == (Dependency("g", expected),)
    assert pcc.pipeline.dependency_hashes(caller, {callee: open_callee(callee)}) == (Dependency("g", expected),)
    assert pcc.pipeline.dependency_hashes(caller, {}) == ()


def test_dependency_hashes_reach_certificate_from_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()
    caller = f_ref()
    callees = {callee: proven_callee(callee)}

    result = _run(caller, tmp_path, callees)

    hashes = pcc.pipeline.dependency_hashes(caller, callees)
    certificate = from_result(result, Config(), dependency_hashes=hashes)

    assert certificate.dependency_hashes == hashes
    assert certificate.assumptions == ()


def test_open_callee_without_translation_lists_assumption_without_dafny(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()

    result = _run(f_ref(), tmp_path, {callee: make_result(callee, State.OPEN, reason="no proof found")})

    assert result.state is State.PROVEN
    assert len(result.assumptions) == 1
    assumption = result.assumptions[0]
    assert assumption.kind is AssumptionKind.UNPROVEN_CALLEE
    assert assumption.subject == "g"
    assert assumption.dafny == ""
    assert recorded.prove_extra == [()]
    assert recorded.prove_imports == [()]
    assert recorded.prove_assumptions_module == [""]


def test_ai_fallback_still_passes_callee_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    recorded.transpile_fails = True
    _install_steps(monkeypatch, recorded)
    callee = g_ref()

    result = _run(f_ref(), tmp_path, {callee: proven_callee(callee)})

    assert result.state is State.PROVEN
    assert result.translation is not None
    assert result.translation.function_path is TranslationPath.AI
    assert recorded.prove_extra == [(G_PROOF_SOURCE,)]
    assert recorded.prove_imports == [(module_name(callee),)]


def test_failed_proof_searches_counterexample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    recorded.prove_outcome = ProofOutcome(
        False,
        CALLER_PROOF_SOURCE,
        2,
        "spark",
        "assertion might not hold",
        (DafnyError(3, 1, "assertion might not hold", "spec(x, result)"),),
    )
    recorded.counterexample = make_counterexample()
    _install_steps(monkeypatch, recorded)

    result = _run(f_ref(), tmp_path, {})

    assert result.state is State.COUNTEREXAMPLE
    assert result.counterexample == recorded.counterexample
    assert result.second_opinion is not None


def test_failed_proof_without_counterexample_stays_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    recorded.prove_outcome = ProofOutcome(False, CALLER_PROOF_SOURCE, 2, "spark", "assertion might not hold", ())
    recorded.counterexample = None
    _install_steps(monkeypatch, recorded)

    result = _run(f_ref(), tmp_path, {})

    assert result.state is State.OPEN
    assert result.reason == "assertion might not hold"


def _open_callee_result(ref: FunctionRef, reason: str, dafny_function: str = G_DAFNY_FUNCTION) -> Result:
    """An unproven callee result with a translated method and spec."""
    return Result(
        ref,
        State.OPEN,
        reason,
        make_intent(ref),
        make_spec(ref),
        make_translation(ref, dafny_function=dafny_function, dafny_spec=G_DAFNY_SPEC),
        (),
        None,
        None,
    )


def test_visible_cycle_gates_caller_without_mutual_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("steps must not run for a visible mutual cycle")

    monkeypatch.setattr(pcc.pipeline, "collect", fail)
    monkeypatch.setattr(pcc.pipeline, "prove", fail)
    callee_a = a_ref()
    callee_b = b_ref()

    result = _run(
        fa_ref(),
        tmp_path,
        {
            callee_a: _open_callee_result(callee_a, "no proof found"),
            callee_b: _open_callee_result(callee_b, "no proof found"),
        },
    )

    assert result.state is State.OPEN
    assert result.reason == "mutual recursion"


TRAILING_COMMENT_FUNCTION = (
    "method g(x: int) returns (result: int)\n"
    "  requires pre(x)\n"
    "  ensures spec(x, result)\n"
    "{\n"
    "  result := x;\n"
    "}\n"
    "// trailing note\n"
)


def test_open_callee_with_trailing_comment_after_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()

    result = _run(f_ref(), tmp_path, {callee: _open_callee_result(callee, "no proof found", TRAILING_COMMENT_FUNCTION)})

    assert result.state is State.PROVEN
    (assumption,) = result.assumptions
    assert assumption.dafny.count("{:axiom}") == 1
    assert assumption.dafny.startswith("method {:axiom} g(")
    assert "requires pre(x)" in assumption.dafny


LEADING_COMMENT_FUNCTION = (
    "// method sketch for g\n"
    "method g(x: int) returns (result: int)\n"
    "  requires pre(x)\n"
    "  ensures spec(x, result)\n"
    "{\n"
    "  result := x;\n"
    "}\n"
)


def test_open_callee_with_method_word_in_comment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _Recorded()
    _install_steps(monkeypatch, recorded)
    callee = g_ref()

    result = _run(f_ref(), tmp_path, {callee: _open_callee_result(callee, "no proof found", LEADING_COMMENT_FUNCTION)})

    assert result.state is State.PROVEN
    (assumption,) = result.assumptions
    assert assumption.dafny.count("{:axiom}") == 1
    assert assumption.dafny.split("\n")[0] == "// method sketch for g"
    assert "method {:axiom} g(x: int)" in assumption.dafny
    assert "requires pre(x)" in assumption.dafny


H_CALLER_SOURCE = "def fh(x: int) -> int:\n    return g(x) + h(x)\n"


def test_first_deciding_callee_in_path_order_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("steps must not run after a deciding callee")

    monkeypatch.setattr(pcc.pipeline, "collect", fail)
    monkeypatch.setattr(pcc.pipeline, "prove", fail)
    callee_g = g_ref()
    callee_h = function_ref(source=H_SOURCE, path=PATH, qualname="h", lineno=3)
    caller = function_ref(source=H_CALLER_SOURCE, path=PATH, qualname="fh", lineno=9)

    first_counterexample = _run(
        caller,
        tmp_path,
        {
            callee_g: make_result(callee_g, State.COUNTEREXAMPLE),
            callee_h: make_result(callee_h, State.NOT_CHECKABLE),
        },
    )
    assert first_counterexample.state is State.OPEN
    assert first_counterexample.reason == "callee g has a counterexample"

    first_not_checkable = _run(
        caller,
        tmp_path,
        {
            callee_g: make_result(callee_g, State.NOT_CHECKABLE),
            callee_h: make_result(callee_h, State.COUNTEREXAMPLE),
        },
    )
    assert first_not_checkable.state is State.NOT_CHECKABLE
    assert first_not_checkable.reason == "callee g not checkable"


def test_dependency_hashes_agree_with_cli_stored_hashes() -> None:
    from pcc.cli import _hashes_for_key

    callee = g_ref()
    caller = f_ref()
    key = (caller.path, caller.qualname)
    by_key: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {key: ((callee.path, callee.qualname),)}
    specs = {(callee.path, callee.qualname): G_DAFNY_SPEC}

    assert _hashes_for_key(key, by_key, specs) == pcc.pipeline.dependency_hashes(caller, {callee: proven_callee(callee)})


def test_run_many_serializes_model_calls() -> None:
    import threading
    import time

    from pcc.models import Completion, Message, ModelClient

    active = 0
    max_active = 0
    guard = threading.Lock()

    class _SlowClient:
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
            nonlocal active, max_active
            _ = (model, messages, step, schema, temperature, max_tokens)
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return Completion("{}", Usage("prove", None, 0, 0, 0.0, 0.0))

    wrapper: ModelClient = pcc.pipeline.SerializedClient(_SlowClient())
    barrier = threading.Barrier(8)
    failures: list[BaseException] = []

    def call() -> None:
        try:
            barrier.wait(timeout=10)
            wrapper.complete("model", (), step="prove")
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert max_active == 1
