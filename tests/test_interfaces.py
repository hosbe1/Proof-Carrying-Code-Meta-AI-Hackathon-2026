"""Contract test for the scaffold signatures of issue #6.

Every stub listed in the issue exists with the stated parameter names, kinds and defaults, and raises
``NotImplementedError``. Bodies are filled in by later step tickets; this test locks the signatures they call.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, get_type_hints

import pytest

import pcc.certificate
import pcc.core
import pcc.counterexample
import pcc.equivalence
import pcc.models
import pcc.pipeline
import pcc.project
import pcc.project.calls as project_calls
import pcc.project.diff as project_diff
import pcc.prove
import pcc.prove.agent as prove_agent
import pcc.pylib
import pcc.report
import pcc.sandbox
import pcc.sandbox.loader as sandbox_loader
import pcc.sandbox.strategies as sandbox_strategies
import pcc.spec
import pcc.translate_ai
import pcc.transpile
import pcc.transpile.code as transpile_code
import pcc.transpile.spec as transpile_spec_module
import pcc.verifier
import pcc.verifier.dafny as verifier_dafny
import pcc.verifier.ghost as verifier_ghost
import pcc.verifier.layout as verifier_layout
from pcc.certificate import store as certificate_store
from pcc.config import Config
from pcc.core import (
    Assumption,
    Certificate,
    CheckKind,
    CheckRecord,
    Counterexample,
    Dependency,
    FunctionRef,
    ProofOutcome,
    Result,
    Spec,
    Translation,
)
from pcc.models import Call, Completion, FakeClient, Message, ModelClient, Router
from pcc.models import router as models_router

# Submodules whose name collides with a re-exported stub of the same package: ``import ... as`` would bind the
# stub, so they are resolved through the module cache instead.
verifier_anti_cheat = importlib.import_module("pcc.verifier.anti_cheat")
spec_write = importlib.import_module("pcc.spec.write")
spec_check = importlib.import_module("pcc.spec.check")
equivalence_spec_check = importlib.import_module("pcc.equivalence.spec_check")
translate_ai_translate = importlib.import_module("pcc.translate_ai.translate")
counterexample_search = importlib.import_module("pcc.counterexample.search")
counterexample_build = importlib.import_module("pcc.counterexample.build")
report_render = importlib.import_module("pcc.report.render")

P = inspect.Parameter
POS = P.POSITIONAL_OR_KEYWORD
KWO = P.KEYWORD_ONLY
EMPTY = P.empty

NoDefault = EMPTY


def assert_sig(func: Any, params: list[tuple[str, Any, Any]], returns: Any) -> inspect.Signature:
    """Assert parameter names, kinds and defaults of ``func`` plus its return annotation."""
    sig = inspect.signature(func)
    actual = [(name, p.kind, p.default) for name, p in sig.parameters.items()]
    assert [(name, kind) for name, kind, _ in actual] == [(name, kind) for name, kind, _ in params]
    for (name, _, expected_default), (_, _, actual_default) in zip(params, actual, strict=True):
        if expected_default is NoDefault:
            assert actual_default is EMPTY, name
        else:
            assert actual_default == expected_default, name
            assert type(actual_default) is type(expected_default), name
    assert get_type_hints(func).get("return") == returns
    return sig


def assert_raises(func: Any, sig: inspect.Signature) -> None:
    """Call a stub with dummy values; the body must raise before touching them."""
    args: list[Any] = [object() for p in sig.parameters.values() if p.kind in (P.POSITIONAL_ONLY, POS)]
    kwargs: dict[str, Any] = {
        name: object() for name, p in sig.parameters.items() if p.kind is KWO and p.default is EMPTY
    }
    with pytest.raises(NotImplementedError):
        func(*args, **kwargs)


def test_core_additive_fields() -> None:
    assert CheckKind.CANDIDATE_FIX.value == "candidate_fix"
    assert CheckKind("candidate_fix") is CheckKind.CANDIDATE_FIX
    assert dataclasses.is_dataclass(Dependency)
    assert [f.name for f in dataclasses.fields(Dependency)] == ["name", "sha256"]
    assert Dependency("pcc.pylib", "abc").sha256 == "abc"
    with pytest.raises(dataclasses.FrozenInstanceError):
        Dependency("n", "s").name = "other"  # type: ignore[misc]
    assert Certificate.__dataclass_fields__["dependency_hashes"].default == ()
    assert CheckRecord.__dataclass_fields__["inputs_total"].default is None
    assert Counterexample.__dataclass_fields__["suggested_fix"].default is None
    assert Spec.__dataclass_fields__["precondition_sources"].default == ()


def test_model_client_complete() -> None:
    assert_sig(
        ModelClient.complete,
        [
            ("self", POS, NoDefault),
            ("model", POS, NoDefault),
            ("messages", POS, NoDefault),
            ("step", KWO, NoDefault),
            ("schema", KWO, None),
            ("temperature", KWO, 0.0),
            ("max_tokens", KWO, None),
        ],
        Completion,
    )


def test_fake_client_complete_and_calls() -> None:
    sig = assert_sig(
        FakeClient.complete,
        [
            ("self", POS, NoDefault),
            ("model", POS, NoDefault),
            ("messages", POS, NoDefault),
            ("step", KWO, NoDefault),
            ("schema", KWO, None),
            ("temperature", KWO, 0.0),
            ("max_tokens", KWO, None),
        ],
        Completion,
    )
    assert isinstance(sig, inspect.Signature)
    assert [f.name for f in dataclasses.fields(Call)] == [
        "model",
        "messages",
        "step",
        "schema",
        "temperature",
        "max_tokens",
    ]
    hints = get_type_hints(Call)
    assert hints["model"] is str
    assert hints["messages"] == tuple[Message, ...]
    assert hints["step"] is str
    assert hints["temperature"] is float
    call = Call("m", (Message("user", "hi"),), "spec")
    assert call.schema is None
    assert call.temperature == 0.0
    assert call.max_tokens is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        call.model = "other"  # type: ignore[misc]


def test_router_and_available_stages() -> None:
    assert isinstance(Router(), Router)
    sig = assert_sig(
        models_router.available_stages,
        [("config", POS, NoDefault), ("router", POS, NoDefault)],
        Config,
    )
    assert_raises(models_router.available_stages, sig)


def test_verifier_signatures() -> None:
    from pcc.core import DafnyError
    from pcc.verifier import VerifyOutcome, anti_cheat, build_python, verify

    assert verifier_anti_cheat.anti_cheat is anti_cheat
    assert verifier_dafny.verify is verify
    assert verifier_dafny.build_python is build_python

    # The anti-cheat body is implemented, so only its signature stays locked here.
    assert_sig(
        anti_cheat,
        [
            ("dafny_code", POS, NoDefault),
            ("frozen_spec", POS, NoDefault),
            ("assumptions_module", KWO, ""),
        ],
        tuple[str, ...],
    )

    # The layout bodies are implemented, so only their signatures stay locked here.
    assert_sig(verifier_layout.method_name, [("function", POS, NoDefault)], str)
    assert_sig(verifier_layout.module_name, [("function", POS, NoDefault)], str)
    assert_sig(
        verifier_layout.assemble,
        [
            ("function", POS, NoDefault),
            ("dafny_spec", POS, NoDefault),
            ("dafny_function", POS, NoDefault),
            ("lemmas", KWO, ""),
            ("imports", KWO, ()),
        ],
        str,
    )

    # resolve is implemented (issue #18); only its signature stays locked here.
    assert_sig(
        verifier_dafny.resolve,
        [("dafny_code", POS, NoDefault), ("timeout_seconds", KWO, 120.0)],
        tuple[DafnyError, ...],
    )
    # verify is implemented (issue #20); only its signature stays locked here.
    assert_sig(
        verify,
        [
            ("dafny_code", POS, NoDefault),
            ("assumptions_module", KWO, ""),
            ("resource_limit", KWO, NoDefault),
            ("timeout_seconds", KWO, None),
            ("extra_modules", KWO, ()),
        ],
        VerifyOutcome,
    )
    # build_python is implemented (issue #29); only its signature stays locked here.
    assert_sig(
        build_python,
        [("dafny_code", POS, NoDefault), ("out_dir", POS, NoDefault)],
        Path,
    )

    # The ghost bodies are implemented, so only their signatures stay locked here.
    assert_sig(
        verifier_ghost.executable_part,
        [("dafny_method", POS, NoDefault), ("lemma_names", KWO, NoDefault)],
        str,
    )
    assert_sig(
        verifier_ghost.same_executable,
        [
            ("original", POS, NoDefault),
            ("proven", POS, NoDefault),
            ("lemma_names", KWO, NoDefault),
        ],
        bool,
    )


def test_sandbox_signatures() -> None:
    from pcc.sandbox import RunOutcome, run_python

    assert sandbox_loader.target_loader is pcc.sandbox.target_loader
    assert sandbox_strategies.strategies_source is pcc.sandbox.strategies_source
    # target_loader and run_python are implemented (issue #9); only their signatures are locked here.
    assert_sig(sandbox_loader.target_loader, [("function", POS, NoDefault)], str)
    # strategies_source is implemented (issue #17); only its signature is locked here.
    assert_sig(sandbox_strategies.strategies_source, [("function", POS, NoDefault)], str)
    assert_sig(
        run_python,
        [
            ("script", POS, NoDefault),
            ("cwd", KWO, NoDefault),
            ("timeout_seconds", KWO, 60.0),
            ("memory_mb", KWO, 1024),
        ],
        RunOutcome,
    )


def test_pylib_signatures() -> None:
    assert callable(pcc.pylib.signatures)
    assert "signatures" in pcc.pylib.__all__
    # signatures is implemented (issue #11); only its signature is locked here.
    assert_sig(pcc.pylib.signatures, [], str)


def test_project_signatures() -> None:
    from pcc.project import affected, call_order, discover

    assert project_diff.changed is pcc.project.changed
    assert project_calls.call_graph is pcc.project.call_graph
    # ``changed`` and ``call_graph`` are implemented (issue #22); only their signatures stay locked here.
    assert_sig(
        pcc.project.changed,
        [("repo_root", POS, NoDefault), ("base", POS, NoDefault)],
        tuple[FunctionRef, ...],
    )
    assert_sig(
        pcc.project.call_graph,
        [("repo_root", POS, NoDefault), ("functions", POS, NoDefault)],
        Mapping[FunctionRef, frozenset[FunctionRef]],
    )
    assert callable(discover)
    assert callable(affected)
    assert callable(call_order)


def test_spec_signatures() -> None:
    assert spec_write.write is pcc.spec.write
    assert spec_check.check is pcc.spec.check
    assert spec_check.spec_cases is pcc.spec.spec_cases
    # write is implemented (issue #26); only its signature is locked here.
    assert_sig(
        pcc.spec.write,
        [
            ("intent", POS, NoDefault),
            ("client", KWO, NoDefault),
            ("model", KWO, NoDefault),
            ("hint", KWO, ""),
        ],
        Spec,
    )
    # check is implemented (issue #54); only its signature is locked here.
    assert_sig(
        pcc.spec.check,
        [
            ("spec", POS, NoDefault),
            ("function", POS, NoDefault),
            ("repo_root", POS, NoDefault),
            ("client", KWO, NoDefault),
            ("model", KWO, NoDefault),
            ("config", KWO, NoDefault),
        ],
        Spec,
    )
    # spec_cases is implemented (issue #54); only its signature is locked here.
    assert_sig(
        pcc.spec.spec_cases,
        [
            ("spec", POS, NoDefault),
            ("function", POS, NoDefault),
            ("repo_root", POS, NoDefault),
            ("config", KWO, NoDefault),
        ],
        tuple[tuple[str, str], ...],
    )


def test_equivalence_signatures() -> None:
    assert equivalence_spec_check.check_spec is pcc.equivalence.check_spec
    assert callable(pcc.equivalence.check)
    sig = assert_sig(
        pcc.equivalence.check_spec,
        [
            ("spec", POS, NoDefault),
            ("dafny_spec", POS, NoDefault),
            ("function", POS, NoDefault),
            ("repo_root", POS, NoDefault),
            ("cases", KWO, NoDefault),
        ],
        CheckRecord,
    )
    assert_raises(pcc.equivalence.check_spec, sig)


def test_transpile_signatures() -> None:
    assert transpile_code.transpile_function is pcc.transpile.transpile_function
    assert transpile_spec_module.transpile_spec is pcc.transpile.transpile_spec
    assert_sig(
        pcc.transpile.transpile_function,
        [("function", POS, NoDefault), ("callees", KWO, MappingProxyType({}))],
        str,
    )
    # transpile_function is implemented (issue #23); only its signature is locked here.
    assert pcc.transpile.translate_expr is transpile_code.translate_expr
    # transpile_spec is implemented (issue #30), so only its signature stays locked here.
    assert_sig(transpile_spec_module.transpile_spec, [("spec", POS, NoDefault)], str)


def test_translate_ai_signatures() -> None:
    assert translate_ai_translate.translate is pcc.translate_ai.translate
    assert_sig(
        pcc.translate_ai.translate,
        [
            ("function", POS, NoDefault),
            ("spec", POS, NoDefault),
            ("client", KWO, NoDefault),
            ("model", KWO, NoDefault),
            ("dafny_spec", KWO, NoDefault),
            ("mismatch", KWO, None),
            ("resolve_retries", KWO, 2),
        ],
        Translation,
    )
    # translate_ai is implemented (issue #35); only its signature is locked here.


def test_counterexample_signatures() -> None:
    assert counterexample_search.search is pcc.counterexample.search
    assert counterexample_build.build is pcc.counterexample.build
    # search is implemented (issue #33); only its signature is locked here.
    assert_sig(
        pcc.counterexample.search,
        [
            ("function", POS, NoDefault),
            ("spec", POS, NoDefault),
            ("repo_root", POS, NoDefault),
            ("failed_clause", KWO, NoDefault),
            ("client", KWO, NoDefault),
            ("model", KWO, NoDefault),
            ("config", KWO, NoDefault),
        ],
        Counterexample | None,
    )
    # build is implemented (issue #33); only its signature is locked here.
    assert_sig(
        pcc.counterexample.build,
        [
            ("function", POS, NoDefault),
            ("spec", POS, NoDefault),
            ("repo_root", POS, NoDefault),
            ("arguments", POS, NoDefault),
        ],
        Counterexample,
    )


def test_prove_signatures() -> None:
    assert prove_agent.prove is pcc.prove.prove
    assert_sig(
        pcc.prove.prove,
        [
            ("translation", POS, NoDefault),
            ("assumptions", POS, NoDefault),
            ("client", KWO, NoDefault),
            ("stages", KWO, NoDefault),
            ("resource_limit", KWO, NoDefault),
            ("function_lines", KWO, NoDefault),
            ("extra_modules", KWO, ()),
            ("assumptions_module", KWO, ""),
        ],
        ProofOutcome,
    )
    # prove is implemented (issue #31); only its signature is locked here.
    hints = get_type_hints(pcc.prove.prove)
    assert hints["assumptions"] == Sequence[Assumption]
    assert hints["translation"] is Translation


def test_certificate_signatures() -> None:
    assert certificate_store.from_result is pcc.certificate.from_result
    assert certificate_store.is_stale is pcc.certificate.is_stale
    assert certificate_store.write is pcc.certificate.write
    assert certificate_store.load is pcc.certificate.load
    assert_sig(
        pcc.certificate.from_result,
        [
            ("result", POS, NoDefault),
            ("config", POS, NoDefault),
            ("dependency_hashes", KWO, ()),
        ],
        Certificate,
    )
    assert_sig(
        pcc.certificate.is_stale,
        [
            ("certificate", POS, NoDefault),
            ("function", POS, NoDefault),
            ("dependency_hashes", KWO, ()),
            ("pylib_sha256", KWO, NoDefault),
        ],
        bool,
    )
    assert callable(pcc.certificate.write)
    assert callable(pcc.certificate.load)


def test_report_signatures() -> None:
    assert report_render.render is pcc.report.render
    # ``render`` is implemented, so only its signature is locked here; its behavior is tested in ``tests/report/``.
    assert_sig(pcc.report.render, [("results", POS, NoDefault), ("stale", KWO, ())], str)


def test_pipeline_signatures() -> None:
    sig = assert_sig(
        pcc.pipeline.run,
        [
            ("function", POS, NoDefault),
            ("repo_root", POS, NoDefault),
            ("client", KWO, NoDefault),
            ("config", KWO, NoDefault),
            ("callees", KWO, MappingProxyType({})),
        ],
        Result,
    )
    assert_raises(pcc.pipeline.run, sig)
    sig = assert_sig(
        pcc.pipeline.run_many,
        [
            ("functions", POS, NoDefault),
            ("repo_root", POS, NoDefault),
            ("client", KWO, NoDefault),
            ("config", KWO, NoDefault),
        ],
        tuple[Result, ...],
    )
    assert_raises(pcc.pipeline.run_many, sig)


def test_all_stubs_are_exported() -> None:
    assert "Call" in pcc.models.__all__
    assert "FakeClient" in pcc.models.__all__
    assert "Router" in pcc.models.__all__
    assert "available_stages" in pcc.models.__all__
    assert "Dependency" in pcc.core.__all__
    assert "Intent" in pcc.core.__all__
    assert "Claim" in pcc.core.__all__
    for module in (
        pcc.verifier,
        pcc.sandbox,
        pcc.project,
        pcc.spec,
        pcc.equivalence,
        pcc.transpile,
        pcc.translate_ai,
        pcc.counterexample,
        pcc.prove,
        pcc.certificate,
        pcc.report,
    ):
        assert isinstance(module.__all__, list)
        for name in module.__all__:
            assert callable(getattr(module, name)), (module.__name__, name)
