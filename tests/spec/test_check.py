"""Step 3, checks: the spec against the original code (Hypothesis) and against mutants."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from pcc.config import SpecConfig
from pcc.core import CheckKind, CheckRecord, Spec, ToolError
from pcc.models import Completion, FakeClient, Message, ModelClient
from pcc.spec import check, spec_cases
from pcc.spec.check import SCHEMA
from tests.helpers import function_ref

CORRECT_PALINDROME = (
    "def is_palindrome(s: str) -> bool:\n"
    "    if len(s) <= 1:\n"
    "        return True\n"
    "    return s[0] == s[-1] and is_palindrome(s[1:-1])\n"
)
BUGGY_PALINDROME = (
    "def is_palindrome(s: str) -> bool:\n"
    "    if len(s) <= 2:\n"
    "        return True\n"
    "    return s[0] == s[-1] and is_palindrome(s[1:-1])\n"
)
PALINDROME_PREDICATE = "def spec(s, result):\n    return result == (s == s[::-1])\n"

ADD_SOURCE = "def add(a: int, b: int) -> int:\n    return a + b\n"
ADD_PREDICATE = "def spec(a, b, result):\n    return result == a + b\n"
ADD_VACUOUS = "def spec(a, b, result):\n    return True\n"
ADD_NONNEG_PRE = "def pre(a, b):\n    return a >= 0\n"
ADD_NEVER_PRE = "def pre(a, b):\n    return False\n"

HEAD_SOURCE = "def head(xs: list[int]) -> int:\n    return xs[0]\n"
HEAD_PREDICATE = "def spec(xs, result):\n    return result == xs[0]\n"

DEAD_SOURCE = "def pick(x: int) -> int:\n    y = x + 1\n    return x\n"
DEAD_PREDICATE = "def spec(x, result):\n    return result == x\n"


def write_module(repo_root: Path, relative: str, source: str) -> None:
    """Write ``source`` to ``repo_root / relative``, creating parent packages with ``__init__.py``."""
    path = repo_root / relative
    for parent in path.parents:
        if parent == repo_root:
            break
        parent.mkdir(parents=True, exist_ok=True)
        init = parent / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    path.write_text(source, encoding="utf-8")


def make_config(**overrides: object) -> SpecConfig:
    """A small ``SpecConfig`` so the sandbox runs stay fast."""
    fields: dict[str, object] = {
        "hypothesis_examples": 50,
        "mutant_inputs": 10,
        "max_rule_mutants": 10,
        "model_mutants": 0,
        "min_kill_ratio": 0.9,
        "max_rewrites": 2,
    }
    fields.update(overrides)
    return SpecConfig(**fields)  # type: ignore[arg-type]


def records_of(result: Spec, kind: CheckKind) -> tuple[CheckRecord, ...]:
    """The check records of ``kind`` in ``result``, in order."""
    return tuple(record for record in result.checks if record.kind is kind)


def prompt_of(client: FakeClient) -> str:
    """The concatenated prompt of the single model call the client recorded."""
    assert len(client.calls) == 1
    return "\n".join(message.content for message in client.calls[0].messages if isinstance(message, Message))


def test_buggy_palindrome_hypothesis_fails_with_shrunk_input(tmp_path: Path) -> None:
    write_module(tmp_path, "strings/palindrome.py", BUGGY_PALINDROME)
    function = function_ref(source=BUGGY_PALINDROME, path="strings/palindrome.py", qualname="is_palindrome")
    spec = Spec(function=function, predicate=PALINDROME_PREDICATE)
    client = FakeClient([json.dumps({"mutants": []})])

    result = check(
        spec,
        function,
        tmp_path,
        client=client,
        model="test-model",
        config=make_config(model_mutants=2, mutant_inputs=40, max_rule_mutants=15),
    )

    kinds = [record.kind for record in result.checks]
    assert kinds[0] is CheckKind.HYPOTHESIS_ORIGINAL
    assert kinds[1] is CheckKind.MUTANTS
    (hyp,) = records_of(result, CheckKind.HYPOTHESIS_ORIGINAL)
    assert hyp.passed is False
    shrunk = ast.literal_eval(hyp.detail)
    assert isinstance(shrunk, str) and len(shrunk) == 2 and shrunk[0] != shrunk[1]
    assert hyp.inputs_tried is not None and hyp.inputs_tried > 0
    assert hyp.inputs_total == 50


def test_buggy_palindrome_mutants_pass_with_candidate_fix(tmp_path: Path) -> None:
    write_module(tmp_path, "strings/palindrome.py", BUGGY_PALINDROME)
    function = function_ref(source=BUGGY_PALINDROME, path="strings/palindrome.py", qualname="is_palindrome")
    spec = Spec(function=function, predicate=PALINDROME_PREDICATE)
    client = FakeClient([json.dumps({"mutants": []})])

    result = check(
        spec,
        function,
        tmp_path,
        client=client,
        model="test-model",
        config=make_config(model_mutants=2, mutant_inputs=40, max_rule_mutants=15),
    )

    (mutants,) = records_of(result, CheckKind.MUTANTS)
    assert mutants.passed is True
    fixes = records_of(result, CheckKind.CANDIDATE_FIX)
    assert fixes
    assert all(fix.passed is True for fix in fixes)
    assert any("len(s) <= 1" in fix.detail for fix in fixes)
    assert len(client.calls) == 1
    assert client.calls[0].step == "mutants"
    assert client.calls[0].schema == SCHEMA
    assert BUGGY_PALINDROME in prompt_of(client)


def test_vacuous_spec_mutants_fail(tmp_path: Path) -> None:
    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_VACUOUS)
    client = FakeClient([])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config())

    (hyp,) = records_of(result, CheckKind.HYPOTHESIS_ORIGINAL)
    assert hyp.passed is True
    (mutants,) = records_of(result, CheckKind.MUTANTS)
    assert mutants.passed is False
    assert "-> original " in mutants.detail and ", mutant " in mutants.detail
    assert records_of(result, CheckKind.CANDIDATE_FIX) == ()


def test_mutants_detail_never_contains_mutant_source(tmp_path: Path) -> None:
    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_VACUOUS)
    client = FakeClient([])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config())

    (mutants,) = records_of(result, CheckKind.MUTANTS)
    assert mutants.passed is False
    assert "return a - b" not in mutants.detail


def test_precondition_never_holds(tmp_path: Path) -> None:
    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_PREDICATE, precondition=ADD_NEVER_PRE)
    client = FakeClient([])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config())

    (hyp,) = records_of(result, CheckKind.HYPOTHESIS_ORIGINAL)
    assert hyp.passed is False
    assert hyp.detail == "precondition never holds"
    assert hyp.inputs_tried == 0
    (mutants,) = records_of(result, CheckKind.MUTANTS)
    assert mutants.passed is False
    assert mutants.detail == "precondition never holds"
    assert client.calls == []


def test_original_raise_reports_exception_type(tmp_path: Path) -> None:
    write_module(tmp_path, "header.py", HEAD_SOURCE)
    function = function_ref(source=HEAD_SOURCE, path="header.py", qualname="head", lineno=1)
    spec = Spec(function=function, predicate=HEAD_PREDICATE)
    client = FakeClient([json.dumps({"mutants": []})])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config(model_mutants=2))

    (hyp,) = records_of(result, CheckKind.HYPOTHESIS_ORIGINAL)
    assert hyp.passed is False
    assert hyp.detail == "[] raises IndexError"


def test_inputs_tried_counts_pre_holding(tmp_path: Path) -> None:
    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_PREDICATE, precondition=ADD_NONNEG_PRE)
    client = FakeClient([])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config())

    (hyp,) = records_of(result, CheckKind.HYPOTHESIS_ORIGINAL)
    assert hyp.passed is True
    assert hyp.inputs_total == 50
    assert hyp.inputs_tried is not None
    assert 0 < hyp.inputs_tried < 50


def test_inputs_counts_equal_without_pre(tmp_path: Path) -> None:
    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_PREDICATE)
    client = FakeClient([])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config())

    (hyp,) = records_of(result, CheckKind.HYPOTHESIS_ORIGINAL)
    assert hyp.passed is True
    assert hyp.inputs_total == 50
    assert hyp.inputs_tried == 50


def test_model_mutant_killed_counts_towards_ratio(tmp_path: Path) -> None:
    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_PREDICATE)
    variant = "def add(a: int, b: int) -> int:\n    return a - b\n"
    client = FakeClient([json.dumps({"mutants": [variant]})])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config(model_mutants=2))

    (mutants,) = records_of(result, CheckKind.MUTANTS)
    assert mutants.passed is True
    assert len(client.calls) == 1
    assert client.calls[0].step == "mutants"
    assert ADD_SOURCE in prompt_of(client)


@pytest.mark.parametrize(
    "answer",
    [
        "not json",
        json.dumps({"other": []}),
        json.dumps({"mutants": "not a list"}),
        json.dumps({"mutants": ["def broken(:"]}),
    ],
)
def test_malformed_model_answer_contributes_zero_mutants(tmp_path: Path, answer: str) -> None:
    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_PREDICATE)
    client = FakeClient([answer])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config(model_mutants=2))

    (mutants,) = records_of(result, CheckKind.MUTANTS)
    assert mutants.passed is True
    assert len(client.calls) == 1


def test_tool_error_from_client_passes_through(tmp_path: Path) -> None:
    class BrokenClient(ModelClient):
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
            raise ToolError("model API down")

    write_module(tmp_path, "adder.py", ADD_SOURCE)
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_PREDICATE)

    with pytest.raises(ToolError, match="model API down"):
        check(
            spec,
            function,
            tmp_path,
            client=BrokenClient(),
            model="test-model",
            config=make_config(model_mutants=2),
        )


def test_equivalent_mutants_dropped_no_mutants(tmp_path: Path) -> None:
    write_module(tmp_path, "picker.py", DEAD_SOURCE)
    function = function_ref(source=DEAD_SOURCE, path="picker.py", qualname="pick", lineno=1)
    spec = Spec(function=function, predicate=DEAD_PREDICATE)
    client = FakeClient([])

    result = check(spec, function, tmp_path, client=client, model="test-model", config=make_config())

    (hyp,) = records_of(result, CheckKind.HYPOTHESIS_ORIGINAL)
    assert hyp.passed is True
    (mutants,) = records_of(result, CheckKind.MUTANTS)
    assert mutants.passed is True
    assert mutants.detail == "no mutants"


def test_spec_cases_deterministic_with_bool_results(tmp_path: Path) -> None:
    write_module(tmp_path, "strings/palindrome.py", CORRECT_PALINDROME)
    function = function_ref(source=CORRECT_PALINDROME, path="strings/palindrome.py", qualname="is_palindrome")
    spec = Spec(function=function, predicate=PALINDROME_PREDICATE)
    config = make_config()

    first = spec_cases(spec, function, tmp_path, config=config)
    second = spec_cases(spec, function, tmp_path, config=config)

    assert first == second
    assert len(first) > 10
    for _, result in first:
        assert isinstance(ast.literal_eval(result), bool)


def test_spec_cases_reports_raises(tmp_path: Path) -> None:
    write_module(tmp_path, "header.py", HEAD_SOURCE)
    function = function_ref(source=HEAD_SOURCE, path="header.py", qualname="head", lineno=1)
    spec = Spec(function=function, predicate=HEAD_PREDICATE)
    config = make_config()

    first = spec_cases(spec, function, tmp_path, config=config)
    second = spec_cases(spec, function, tmp_path, config=config)

    assert first == second
    assert len(first) > 10
    for _, result in first:
        assert result == "raises IndexError" or isinstance(ast.literal_eval(result), int)


def test_missing_repo_root_raises_tool_error(tmp_path: Path) -> None:
    function = function_ref(source=ADD_SOURCE, path="adder.py", qualname="add", lineno=1)
    spec = Spec(function=function, predicate=ADD_PREDICATE)
    client = FakeClient([])

    with pytest.raises(ToolError):
        check(
            spec,
            function,
            tmp_path / "missing",
            client=client,
            model="test-model",
            config=make_config(),
        )
