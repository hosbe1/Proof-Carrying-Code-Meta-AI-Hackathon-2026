"""Tests for the training reward and the shared system message.

The three examples from the ticket (a correct proof, a wrong proof, ``assume false;``) are
``dafny``-marked: they run the real ``verifier.verify``. The order and error rules are pinned
with unit tests that never touch the Dafny CLI.
"""

from __future__ import annotations

import importlib

import pytest
from training.prompts import SYSTEM_MESSAGE
from training.reward import reward

from pcc.config import Config
from pcc.core import ToolError
from pcc.verifier import VerifyOutcome
from training import prompts as prompts_module

# ``training/__init__.py`` re-exports the ``reward`` function, so ``training.reward``
# resolves to the function, not the submodule; go through the module cache instead.
reward_module = importlib.import_module("training.reward")

RESOURCE_LIMIT = Config().dafny.resource_limit

TASK_SPEC = (
    "predicate spec(x: int, result: int) { result == x }\n"
    "method f(x: int) returns (result: int)\n"
    "  ensures spec(x, result)\n"
    "{\n"
)

CORRECT = TASK_SPEC + "  result := x;\n}\n"
WRONG = TASK_SPEC + "  result := x + 1;\n}\n"
CHEAT = TASK_SPEC + "  assume false;\n}\n"
UNRESOLVABLE = TASK_SPEC + "  nosuch := 1;\n}\n"


@pytest.mark.dafny
def test_correct_proof_gives_one() -> None:
    assert reward(CORRECT, TASK_SPEC, resource_limit=RESOURCE_LIMIT) == 1.0


@pytest.mark.dafny
def test_wrong_proof_gives_zero() -> None:
    assert reward(WRONG, TASK_SPEC, resource_limit=RESOURCE_LIMIT) == 0.0


@pytest.mark.dafny
def test_assume_gives_minus_one() -> None:
    assert reward(CHEAT, TASK_SPEC, resource_limit=RESOURCE_LIMIT) == -1.0


@pytest.mark.dafny
def test_unresolvable_answer_gives_zero() -> None:
    assert reward(UNRESOLVABLE, TASK_SPEC, resource_limit=RESOURCE_LIMIT) == 0.0


@pytest.mark.dafny
def test_changed_spec_gives_minus_one() -> None:
    changed = CORRECT.replace("result == x", "result == x + 1")
    assert TASK_SPEC not in changed
    assert reward(changed, TASK_SPEC, resource_limit=RESOURCE_LIMIT) == -1.0


def test_absent_spec_gives_minus_one() -> None:
    assert reward("method g() returns (r: int)\n{\n  r := 0;\n}\n", TASK_SPEC, resource_limit=RESOURCE_LIMIT) == -1.0


def test_empty_spec_gives_minus_one() -> None:
    assert reward(CORRECT, "", resource_limit=RESOURCE_LIMIT) == -1.0


def test_added_requires_gives_minus_one_without_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ticket example 2: an answer with an added requires changes the header, giving -1."""

    def fail_on_verify(_code: str, *, resource_limit: int) -> VerifyOutcome:
        raise AssertionError("verify must not run for a header violation")

    monkeypatch.setattr(reward_module, "verify", fail_on_verify)
    spec = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n {}"
    answer = "method M(n: int) returns (r: int) requires n >= 0 ensures r == n requires false { r := n; }"
    assert reward(answer, spec, resource_limit=RESOURCE_LIMIT) == -1.0


def test_duplicated_method_with_added_requires_gives_minus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frozen prefix still occurs, but the duplicated header differs: -1 without verify."""

    def fail_on_verify(_code: str, *, resource_limit: int) -> VerifyOutcome:
        raise AssertionError("verify must not run for a header violation")

    monkeypatch.setattr(reward_module, "verify", fail_on_verify)
    answer = CORRECT + (
        "method f(x: int) returns (result: int)\n  ensures spec(x, result)\n  requires false\n{\n  result := x;\n}\n"
    )
    assert TASK_SPEC in answer
    assert reward(answer, TASK_SPEC, resource_limit=RESOURCE_LIMIT) == -1.0


def test_tool_error_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_code: str, *, resource_limit: int) -> VerifyOutcome:
        raise ToolError("dafny CLI is not on PATH")

    monkeypatch.setattr(reward_module, "verify", broken)
    with pytest.raises(ToolError, match="dafny CLI"):
        reward(CORRECT, TASK_SPEC, resource_limit=RESOURCE_LIMIT)


def test_resource_limit_reaches_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_verify(_code: str, *, resource_limit: int) -> VerifyOutcome:
        seen.append(resource_limit)
        return VerifyOutcome(True, (), 0.0)

    monkeypatch.setattr(reward_module, "verify", fake_verify)
    assert reward(CORRECT, TASK_SPEC, resource_limit=123) == 1.0
    assert seen == [123]


def test_system_message_is_non_empty_string() -> None:
    assert isinstance(SYSTEM_MESSAGE, str)
    assert SYSTEM_MESSAGE != ""
    assert prompts_module.SYSTEM_MESSAGE is SYSTEM_MESSAGE


def test_system_message_is_the_shared_training_prompt() -> None:
    assert SYSTEM_MESSAGE == (
        "You are a Dafny 4.11 verification assistant. Return only a complete Dafny program. "
        "Preserve the supplied specification and method declaration exactly. "
        "If executable code is supplied, preserve it and add only ghost proof annotations. "
        "Do not use assume, axioms, verification-disabling attributes, or resource-limit overrides. "
        "The program must verify without --allow-warnings."
    )
