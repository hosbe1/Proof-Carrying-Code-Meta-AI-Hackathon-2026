"""Tests for step 4: the AI fallback translation of what the transpiler does not support."""

from __future__ import annotations

import importlib
import json

import pytest

from pcc.core import DafnyError, FunctionRef, NotCheckable, Spec, ToolError, TranslationPath
from pcc.models import Completion, FakeClient, Message
from pcc.pylib import signatures as pylib_signatures
from pcc.translate_ai import translate
from pcc.translate_ai.prompts import SCHEMA
from pcc.verifier import assemble
from tests.helpers import counterexample, function_ref
from tests.helpers import spec as make_spec_helper

translate_module = importlib.import_module("pcc.translate_ai.translate")

FUNCTION_SOURCE = "def maximum(xs: list[int]) -> int:\n    return max(xs)\n"

PYTHON_SPEC = "def spec(xs, result):\n    return result in xs and all(x <= result for x in xs)\n"

PYTHON_PRE = "def pre(xs):\n    return len(xs) > 0\n"

DAFNY_SPEC = (
    "predicate pre(xs: seq<int>)\n{\n  |xs| > 0\n}\n\npredicate spec(xs: seq<int>, result: int)\n{\n  result in xs\n}\n"
)

GOOD_METHOD = (
    "method maximum(xs: seq<int>) returns (result: int)\n"
    "  requires pre(xs)\n"
    "  ensures spec(xs, result)\n"
    "{\n"
    "  result := xs[0];\n"
    "}\n"
)

TYPE_ERROR_METHOD = (
    "method maximum(xs: seq<int>) returns (result: int)\n"
    "  requires pre(xs)\n"
    "  ensures spec(xs, result)\n"
    "{\n"
    '  result := "not an int";\n'
    "}\n"
)

SPEC_CHANGED_METHOD = (
    "predicate spec(xs: seq<int>, result: int)\n"
    "{\n"
    "  true\n"
    "}\n"
    "\n"
    "method maximum(xs: seq<int>) returns (result: int)\n"
    "  requires pre(xs)\n"
    "  ensures spec(xs, result)\n"
    "{\n"
    "  result := xs[0];\n"
    "}\n"
)

EXTRA_REQUIRES_METHOD = (
    "method maximum(xs: seq<int>) returns (result: int)\n"
    "  requires pre(xs)\n"
    "  requires |xs| > 0\n"
    "  ensures spec(xs, result)\n"
    "{\n"
    "  result := xs[0];\n"
    "}\n"
)

RENAMED_ENSURES_METHOD = (
    "method maximum(xs: seq<int>) returns (result: int)\n"
    "  requires pre(xs)\n"
    "  ensures other_spec(xs, result)\n"
    "{\n"
    "  result := xs[0];\n"
    "}\n"
)


def make_function() -> FunctionRef:
    """The ``maximum`` function the AI fallback translates in these tests."""
    return function_ref(source=FUNCTION_SOURCE, path="src/shop.py", qualname="maximum", lineno=3)


def make_spec(function: FunctionRef) -> Spec:
    """The Python spec of ``maximum``, with a precondition."""
    return make_spec_helper(function, predicate=PYTHON_SPEC, precondition=PYTHON_PRE)


def function_answer(method: str) -> str:
    """A model answer with the ``{dafny_function}`` schema."""
    return json.dumps({"dafny_function": method})


def user_prompts(client: FakeClient) -> list[str]:
    """The user message of every model call so far."""
    return ["\n".join(message.content for message in messages) for _, messages, _ in client.requests]


class _ResolveStub:
    """A scripted ``pcc.verifier.resolve`` that records the sources it was given."""

    def __init__(self, outcomes: list[tuple[DafnyError, ...]]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def __call__(self, dafny_code: str, *, timeout_seconds: float = 120.0) -> tuple[DafnyError, ...]:
        _ = timeout_seconds
        self.calls.append(dafny_code)
        assert self._outcomes, "resolve called more often than scripted"
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


def _raising_resolve(dafny_code: str, *, timeout_seconds: float = 120.0) -> tuple[DafnyError, ...]:
    raise ToolError("dafny is broken")


def test_valid_answer_returns_ai_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(GOOD_METHOD)])
    verifier = _ResolveStub([()])
    monkeypatch.setattr(translate_module, "resolve", verifier)

    translation = translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    assert translation.function is function
    assert translation.function_path is TranslationPath.AI
    assert translation.spec_path is TranslationPath.TRANSPILER
    assert translation.dafny_function == GOOD_METHOD
    assert translation.dafny_spec == DAFNY_SPEC
    assert verifier.calls == [assemble(function, DAFNY_SPEC, GOOD_METHOD)]


def test_single_call_uses_translate_ai_step_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(GOOD_METHOD)])
    monkeypatch.setattr(translate_module, "resolve", _ResolveStub([()]))

    translate(function, make_spec(function), client=client, model="muse-spark-1.3", dafny_spec=DAFNY_SPEC)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.step == "translate_ai"
    assert call.model == "muse-spark-1.3"
    assert call.schema == SCHEMA
    assert SCHEMA["required"] == ["dafny_function"]


def test_prompt_holds_source_spec_shape_and_pylib(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(GOOD_METHOD)])
    monkeypatch.setattr(translate_module, "resolve", _ResolveStub([()]))

    translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    prompt = user_prompts(client)[0]
    assert FUNCTION_SOURCE in prompt
    assert PYTHON_SPEC in prompt
    assert PYTHON_PRE in prompt
    assert "maximum" in prompt
    assert DAFNY_SPEC in prompt
    assert pylib_signatures() in prompt
    assert "requires pre" in prompt
    assert "ensures spec" in prompt


def test_type_error_then_valid_takes_two_calls_with_error_in_second_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = make_function()
    client = FakeClient([function_answer(TYPE_ERROR_METHOD), function_answer(GOOD_METHOD)])
    verifier = _ResolveStub([((DafnyError(6, 14, "type error: expected int, got string"),)), ()])
    monkeypatch.setattr(translate_module, "resolve", verifier)

    translation = translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    assert len(client.calls) == 2
    assert len(verifier.calls) == 2
    assert "type error: expected int, got string" in user_prompts(client)[1]
    assert translation.dafny_function == GOOD_METHOD


def test_answer_changing_spec_is_rejected_without_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(SPEC_CHANGED_METHOD), function_answer(GOOD_METHOD)])
    verifier = _ResolveStub([()])
    monkeypatch.setattr(translate_module, "resolve", verifier)

    translation = translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    assert len(client.calls) == 2
    assert len(verifier.calls) == 1
    assert "spec changed" in user_prompts(client)[1]
    assert translation.dafny_function == GOOD_METHOD
    assert translation.dafny_spec == DAFNY_SPEC


def test_answer_with_extra_requires_is_rejected_and_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(EXTRA_REQUIRES_METHOD), function_answer(GOOD_METHOD)])
    verifier = _ResolveStub([()])
    monkeypatch.setattr(translate_module, "resolve", verifier)

    translation = translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    assert len(client.calls) == 2
    assert len(verifier.calls) == 1
    assert "header changed" in user_prompts(client)[1]
    assert translation.dafny_function == GOOD_METHOD


def test_answer_with_renamed_ensures_is_rejected_as_header_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(RENAMED_ENSURES_METHOD), function_answer(GOOD_METHOD)])
    verifier = _ResolveStub([()])
    monkeypatch.setattr(translate_module, "resolve", verifier)

    translation = translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    assert len(client.calls) == 2
    assert len(verifier.calls) == 1
    assert "header changed" in user_prompts(client)[1]
    assert translation.dafny_function == GOOD_METHOD


def test_exhausted_retries_raise_not_checkable(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(TYPE_ERROR_METHOD), function_answer(TYPE_ERROR_METHOD)])
    verifier = _ResolveStub(
        [
            (DafnyError(6, 14, "type error: expected int, got string"),),
            (DafnyError(6, 14, "type error: expected int, got string"),),
        ]
    )
    monkeypatch.setattr(translate_module, "resolve", verifier)

    with pytest.raises(NotCheckable, match="AI translation does not resolve"):
        translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC, resolve_retries=1)

    assert len(client.calls) == 2
    assert len(verifier.calls) == 2


def test_mismatch_text_appears_in_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(GOOD_METHOD)])
    monkeypatch.setattr(translate_module, "resolve", _ResolveStub([()]))

    translate(
        function,
        make_spec(function),
        client=client,
        model="m",
        dafny_spec=DAFNY_SPEC,
        mismatch=counterexample(),
    )

    prompt = user_prompts(client)[0]
    assert "[3, -1], key=None" in prompt
    assert "the maximum of the list" in prompt
    assert "-1" in prompt


def test_non_json_answer_counts_as_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient(["this is not JSON", function_answer(GOOD_METHOD)])
    verifier = _ResolveStub([()])
    monkeypatch.setattr(translate_module, "resolve", verifier)

    translation = translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    assert len(client.calls) == 2
    assert len(verifier.calls) == 1
    assert translation.dafny_function == GOOD_METHOD


def test_answer_without_dafny_function_counts_as_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([json.dumps({"other_key": "x"}), function_answer(GOOD_METHOD)])
    verifier = _ResolveStub([()])
    monkeypatch.setattr(translate_module, "resolve", verifier)

    translation = translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    assert len(client.calls) == 2
    assert translation.dafny_function == GOOD_METHOD


def test_model_tool_error_passes_through() -> None:
    function = make_function()

    with pytest.raises(ToolError, match="model API is down"):
        translate(function, make_spec(function), client=_FailingClient(), model="m", dafny_spec=DAFNY_SPEC)


def test_resolve_tool_error_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    function = make_function()
    client = FakeClient([function_answer(GOOD_METHOD)])
    monkeypatch.setattr(translate_module, "resolve", _raising_resolve)

    with pytest.raises(ToolError, match="dafny is broken"):
        translate(function, make_spec(function), client=client, model="m", dafny_spec=DAFNY_SPEC)

    assert len(client.calls) == 1
