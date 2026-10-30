"""Tests for step 5: the proof agent with escalating prover stages."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

import pcc.prove.agent as prove_agent
from pcc.config import ProverStage
from pcc.core import DafnyError, ToolError, Translation
from pcc.models import Completion, FakeClient, Message
from pcc.prove import prove
from pcc.prove.prompts import SCHEMA
from pcc.verifier import VerifyOutcome, assemble
from tests.helpers import assumption, function_ref, translation

SPEC = (
    "predicate pre(s: string)\n"
    "{\n"
    "  true\n"
    "}\n"
    "\n"
    "predicate spec(s: string, result: bool)\n"
    "{\n"
    "  result == (forall i :: 0 <= i < |s| ==> s[i] == s[|s| - 1 - i])\n"
    "}\n"
)

METHOD = (
    "method is_palindrome(s: string) returns (result: bool)\n"
    "  requires pre(s)\n"
    "  ensures spec(s, result)\n"
    "{\n"
    "  if |s| <= 2 {\n"
    "    result := |s| <= 1 || s[0] == s[|s| - 1];\n"
    "  } else {\n"
    "    var inner := is_palindrome(s[1..|s| - 1]);\n"
    "    result := s[0] == s[|s| - 1] && inner;\n"
    "  }\n"
    "}\n"
)

METHOD_WITH_DECREASES = METHOD.replace(
    "  ensures spec(s, result)\n",
    "  ensures spec(s, result)\n  decreases s\n",
)

BAD_LEMMAS = "lemma Bad() ensures false {}\n"

GOOD_LEMMAS = (
    "lemma PalindromeStep(s: string)\n"
    "  requires |s| > 1\n"
    "  ensures (forall i :: 0 <= i < |s| ==> s[i] == s[|s| - 1 - i]) == "
    "(s[0] == s[|s| - 1] && (forall j :: 0 <= j < |s[1..|s| - 1]| ==> "
    "s[1..|s| - 1][j] == s[1..|s| - 1][|s[1..|s| - 1]| - 1 - j]))\n"
    "{\n"
    "}\n"
)

PROOF_METHOD = METHOD_WITH_DECREASES.replace(
    "    var inner := is_palindrome(s[1..|s| - 1]);\n",
    "    var inner := is_palindrome(s[1..|s| - 1]);\n    PalindromeStep(s);\n",
)


def make_translation() -> Translation:
    """The frozen translation of ``is_palindrome`` the proof agent starts from."""
    function = function_ref(
        source="def is_palindrome(s: str) -> bool:\n    ...",
        path="src/palindrome.py",
        qualname="is_palindrome",
        lineno=3,
    )
    return translation(function, dafny_function=METHOD, dafny_spec=SPEC)


def proof_answer(lemmas: str, method: str) -> str:
    """A model answer with the ``{lemmas, method}`` schema."""
    return json.dumps({"lemmas": lemmas, "method": method})


def user_prompts(client: FakeClient) -> list[str]:
    """The user message of every model call so far."""
    return ["\n".join(message.content for message in messages) for _, messages, _ in client.requests]


class _VerifyStub:
    """A scripted ``pcc.verifier.verify`` that records the sources it was given."""

    def __init__(self, outcomes: list[VerifyOutcome]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def __call__(
        self,
        dafny_code: str,
        *,
        assumptions_module: str = "",
        resource_limit: int,
        timeout_seconds: float | None = None,
        extra_modules: Sequence[str] = (),
    ) -> VerifyOutcome:
        _ = (assumptions_module, resource_limit, timeout_seconds, extra_modules)
        self.calls.append(dafny_code)
        assert self._outcomes, "verify called more often than scripted"
        return self._outcomes.pop(0)


class _FailingClient:
    """A model client that always raises, to check errors pass through."""

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
        raise ToolError("the model API is down")


def test_bad_lemma_then_proof_verifies_on_second_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    client = FakeClient(
        [
            proof_answer(BAD_LEMMAS, METHOD_WITH_DECREASES),
            proof_answer(GOOD_LEMMAS, PROOF_METHOD),
        ]
    )
    verifier = _VerifyStub(
        [
            VerifyOutcome(False, (DafnyError(10, 5, "assertion might not hold", "spec(s, result)"),), 0.1),
            VerifyOutcome(True, (), 0.2),
        ]
    )
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("spark", "m", 2),),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is True
    assert outcome.attempts == 2
    assert outcome.stage == "spark"
    assert outcome.last_error == ""
    assert outcome.errors == ()
    assert outcome.dafny_source == assemble(proven.function, SPEC, PROOF_METHOD, lemmas=GOOD_LEMMAS)
    assert len(verifier.calls) == 2
    assert [request[2] for request in client.requests] == ["proof", "proof"]
    assert [request[0] for request in client.requests] == ["m", "m"]
    assert client.calls[0].schema == SCHEMA
    prompts = user_prompts(client)
    assert SPEC in prompts[0]
    assert METHOD in prompts[0]
    assert "py_floordiv" in prompts[0]
    assert "dafny" in prompts[1]
    assert "assertion might not hold" in prompts[1]


def test_changed_comparison_rejected_without_running_dafny(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    changed = METHOD.replace("|s| <= 2", "|s| <= 1")
    assert changed != METHOD
    client = FakeClient([proof_answer("", changed)])
    verifier = _VerifyStub([])
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("spark", "m", 1),),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is False
    assert outcome.attempts == 1
    assert outcome.stage == "spark"
    assert outcome.last_error == "executable code changed"
    assert verifier.calls == []


def test_changed_requires_rejected_without_running_dafny(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    changed = METHOD.replace("requires pre(s)", "requires pre(s) && |s| > 0")
    client = FakeClient([proof_answer("", changed)])
    verifier = _VerifyStub([])
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("spark", "m", 1),),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is False
    assert outcome.last_error == "executable code changed"
    assert verifier.calls == []


def test_added_modifies_rejected_although_ghost(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    changed = METHOD.replace(
        "  ensures spec(s, result)\n",
        "  ensures spec(s, result)\n  modifies s\n",
    )
    client = FakeClient([proof_answer("", changed)])
    verifier = _VerifyStub([])
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("spark", "m", 1),),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is False
    assert outcome.last_error == "executable code changed"
    assert verifier.calls == []


def test_assume_in_lemmas_rejected_by_anti_cheat(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    lemmas = "lemma Helper(s: string) ensures true\n{\n  assume false;\n}\n"
    client = FakeClient([proof_answer(lemmas, METHOD)])
    verifier = _VerifyStub([])
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("spark", "m", 1),),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is False
    assert outcome.attempts == 1
    assert outcome.last_error == "anti-cheat: assume"
    assert verifier.calls == []


def test_assume_in_method_rejected_as_executable_change(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    changed = METHOD.replace("  if |s| <= 2 {\n", "  if |s| <= 2 {\n    assume false;\n")
    client = FakeClient([proof_answer("", changed)])
    verifier = _VerifyStub([])
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("spark", "m", 1),),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is False
    assert outcome.last_error == "executable code changed"
    assert verifier.calls == []


def test_method_declaration_in_lemmas_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    lemmas = "method Evil() ensures false\n{\n}\n"
    client = FakeClient([proof_answer(lemmas, METHOD)])
    verifier = _VerifyStub([])
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("spark", "m", 1),),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is False
    assert outcome.last_error == "method declaration in lemmas"
    assert verifier.calls == []


def test_all_stages_skipped_when_function_too_large() -> None:
    proven = make_translation()
    client = FakeClient([])

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("wonder", "m", 1, max_source_lines=40),),
        resource_limit=1000,
        function_lines=100,
    )

    assert outcome.proven is False
    assert outcome.attempts == 0
    assert outcome.stage == ""
    assert outcome.last_error == "no prover stage available"
    assert outcome.errors == ()
    assert client.requests == []


def test_skipped_stage_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    client = FakeClient([proof_answer(GOOD_LEMMAS, PROOF_METHOD)])
    verifier = _VerifyStub([VerifyOutcome(True, (), 0.1)])
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(
            ProverStage("wonder", "mw", 1, max_source_lines=40),
            ProverStage("spark", "ms", 1),
        ),
        resource_limit=1000,
        function_lines=100,
    )

    assert outcome.proven is True
    assert outcome.attempts == 1
    assert outcome.stage == "spark"
    assert [request[0] for request in client.requests] == ["ms"]


def test_invalid_json_counts_as_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    client = FakeClient(["{not json", proof_answer(GOOD_LEMMAS, PROOF_METHOD)])
    verifier = _VerifyStub([VerifyOutcome(True, (), 0.1)])
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("spark", "m", 2),),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is True
    assert outcome.attempts == 2
    assert outcome.last_error == ""
    assert len(verifier.calls) == 1
    assert "invalid proof answer" in user_prompts(client)[1]


def test_second_stage_prompt_holds_all_first_stage_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    proven = make_translation()
    client = FakeClient([proof_answer("", METHOD_WITH_DECREASES)] * 3)
    failures = (
        VerifyOutcome(False, (DafnyError(1, 1, "first failure", ""),), 0.1),
        VerifyOutcome(False, (DafnyError(2, 2, "second failure", ""),), 0.1),
        VerifyOutcome(False, (DafnyError(3, 3, "third failure", ""),), 0.1),
    )
    verifier = _VerifyStub(list(failures))
    monkeypatch.setattr(prove_agent, "verify", verifier)

    outcome = prove(
        proven,
        (),
        client=client,
        stages=(ProverStage("glimmer", "m1", 2), ProverStage("spark", "m2", 1)),
        resource_limit=1000,
        function_lines=10,
    )

    assert outcome.proven is False
    assert outcome.attempts == 3
    assert outcome.stage == "spark"
    assert outcome.last_error == "dafny: line 3: third failure"
    assert outcome.errors == failures[2].errors
    assert outcome.dafny_source == assemble(proven.function, SPEC, METHOD_WITH_DECREASES, lemmas="")
    assert [request[0] for request in client.requests] == ["m1", "m1", "m2"]
    prompts = user_prompts(client)
    assert "first failure" in prompts[1]
    assert "second failure" not in prompts[1]
    assert "first failure" in prompts[2]
    assert "second failure" in prompts[2]


def test_non_empty_assumptions_raise_value_error() -> None:
    proven = make_translation()
    client = FakeClient([])

    with pytest.raises(ValueError, match="assumptions"):
        prove(
            proven,
            (assumption(),),
            client=client,
            stages=(ProverStage("spark", "m", 1),),
            resource_limit=1000,
            function_lines=10,
        )
    assert client.requests == []


def test_tool_error_from_client_passes_through() -> None:
    proven = make_translation()

    with pytest.raises(ToolError, match="model API is down"):
        prove(
            proven,
            (),
            client=_FailingClient(),
            stages=(ProverStage("spark", "m", 1),),
            resource_limit=1000,
            function_lines=10,
        )
