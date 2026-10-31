"""Pipeline offline end-to-end on the palindrome (issue #55).

Runs the real steps together on a temporary repository holding one version of
``strings/palindrome.py``. The buggy version (``len(s) <= 2``) ends as a
counterexample, the fixed version (``len(s) <= 1``) ends as proven. No step is
monkeypatched; the only client is a scripted ``FakeClient``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pcc.pipeline
from pcc.certificate import from_result
from pcc.config import (
    Config,
    CounterexampleConfig,
    EquivalenceConfig,
    PipelineConfig,
    ProofConfig,
    ProverStage,
    SpecConfig,
)
from pcc.core import FunctionRef, Result, State
from pcc.models import FakeClient
from pcc.sandbox import run_python
from pcc.verifier import verify
from tests.helpers import function_ref

pytestmark = pytest.mark.dafny

FIXTURES = Path(__file__).parent / "fixtures"

BUGGY_SOURCE = (
    "def is_palindrome_recursive(s: str) -> bool:\n"
    '    """Check whether s reads the same forward and backward."""\n'
    "    if len(s) <= 2:\n"
    "        return True\n"
    "    if s[0] != s[-1]:\n"
    "        return False\n"
    "    return is_palindrome_recursive(s[1:-1])\n"
)

FIXED_SOURCE = BUGGY_SOURCE.replace("len(s) <= 2", "len(s) <= 1")


def _fixture(name: str) -> str:
    """The scripted model answer ``name``."""
    return (FIXTURES / name).read_text(encoding="utf-8")


def _write_module(repo_root: Path, source: str) -> None:
    """Write ``source`` to ``strings/palindrome.py`` under ``repo_root``."""
    target = repo_root / "strings" / "palindrome.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / "__init__.py").write_text("", encoding="utf-8")
    target.write_text(source, encoding="utf-8")


def _make_ref(source: str) -> FunctionRef:
    """The ``FunctionRef`` for the palindrome in the temporary repository."""
    return function_ref(
        source=source,
        path="strings/palindrome.py",
        qualname="is_palindrome_recursive",
        lineno=1,
    )


def _make_config() -> Config:
    """A fast config that still runs every real step on the palindrome.

    ``mutant_inputs`` stays small so the spec-translation cases keep to
    convertible strings; larger input sets hit non-BMP characters the
    back-compiled converter cannot encode (see Risks).
    """
    return Config(
        pipeline=PipelineConfig(search_agent=False),
        spec=SpecConfig(
            hypothesis_examples=50,
            mutant_inputs=4,
            max_rule_mutants=10,
            model_mutants=1,
            min_kill_ratio=0.9,
            max_rewrites=2,
        ),
        equivalence=EquivalenceConfig(
            differential_inputs=200,
            symbolic_seconds=0.0,
            max_retranslations=3,
        ),
        proof=ProofConfig(
            prover_stages=(ProverStage(name="spark", model="muse-spark-1.3", attempts=1),),
        ),
        counterexample=CounterexampleConfig(hypothesis_examples=50, model_inputs=0),
    )


def _run_buggy(repo_root: Path) -> tuple[Result, FakeClient]:
    """Run the pipeline on the buggy palindrome."""
    _write_module(repo_root, BUGGY_SOURCE)
    ref = _make_ref(BUGGY_SOURCE)
    client = FakeClient(
        [
            _fixture("intent.json"),
            _fixture("spec.json"),
            _fixture("mutants.json"),
            _fixture("second_opinion.json"),
        ]
    )
    return pcc.pipeline.run(ref, repo_root, client=client, config=_make_config()), client


def _run_fixed(repo_root: Path) -> tuple[Result, FakeClient]:
    """Run the pipeline on the fixed palindrome."""
    _write_module(repo_root, FIXED_SOURCE)
    ref = _make_ref(FIXED_SOURCE)
    client = FakeClient(
        [
            _fixture("intent.json"),
            _fixture("spec.json"),
            _fixture("mutants.json"),
            _fixture("edge_cases.json"),
            _fixture("proof.json"),
        ]
    )
    return pcc.pipeline.run(ref, repo_root, client=client, config=_make_config()), client


def _generated_test_fails(repo_root: Path, test_source: str) -> bool:
    """Whether the generated counterexample test fails when run in the sandbox."""
    (repo_root / "test_palindrome_breaks.py").write_text(test_source, encoding="utf-8")
    script = (
        "import pytest\n"
        "import sys\n"
        "sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', 'test_palindrome_breaks.py']))\n"
    )
    return run_python(script, cwd=repo_root).returncode != 0


def test_buggy_palindrome_gives_failing_counterexample(tmp_path: Path) -> None:
    result, _ = _run_buggy(tmp_path)
    assert result.state is State.COUNTEREXAMPLE
    assert result.counterexample is not None
    shown = ast.literal_eval(result.counterexample.arguments)
    assert isinstance(shown, str) and len(shown) == 2 and shown[0] != shown[1]
    assert _generated_test_fails(tmp_path, result.counterexample.test)


def test_fixed_palindrome_proves_and_certificate_verifies(tmp_path: Path) -> None:
    result, _ = _run_fixed(tmp_path)
    assert result.state is State.PROVEN
    assert result.proof is not None and result.proof.proven
    config = _make_config()
    certificate = from_result(result, config)
    outcome = verify(certificate.dafny_source, resource_limit=config.dafny.resource_limit)
    assert outcome.verified


def test_end_to_end_uses_real_steps_in_order(tmp_path: Path) -> None:
    buggy_root = tmp_path / "buggy"
    fixed_root = tmp_path / "fixed"
    buggy_root.mkdir()
    fixed_root.mkdir()
    buggy_result, buggy_client = _run_buggy(buggy_root)
    fixed_result, fixed_client = _run_fixed(fixed_root)

    assert buggy_result.state is State.COUNTEREXAMPLE
    assert fixed_result.state is State.PROVEN
    assert [call.step for call in buggy_client.calls] == ["intent", "spec", "mutants", "second_opinion"]
    assert [call.step for call in fixed_client.calls] == ["intent", "spec", "mutants", "edge_cases", "proof"]
    assert "proof" not in [call.step for call in buggy_client.calls]
    assert "counterexample" not in [call.step for call in buggy_client.calls]
    assert "counterexample" not in [call.step for call in fixed_client.calls]
