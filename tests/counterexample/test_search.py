"""Step 6, search: Hypothesis plus model-proposed inputs that break the spec."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcc.config import CounterexampleConfig
from pcc.core import FunctionRef, Spec, ToolError
from pcc.counterexample import search
from pcc.models import Completion, FakeClient, Message, ModelClient
from pcc.sandbox import run_python
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
PALINDROME_CLAUSE = "result == (s == s[::-1])"

GREET_SOURCE = 'def greet(name: str) -> str:\n    return "hi " + name if name != "boom" else "BANG"\n'
GREET_PREDICATE = 'def spec(name, result):\n    return result == "hi " + name\n'

PRE_NONEMPTY = "def pre(s):\n    return len(s) > 0\n"


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


def run_test_file(repo_root: Path, name: str, test_source: str) -> int:
    """Write ``test_source`` to ``name`` under ``repo_root`` and run it with pytest in the sandbox."""
    (repo_root / name).write_text(test_source, encoding="utf-8")
    script = f"import pytest\nimport sys\nsys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', {name!r}]))\n"
    return run_python(script, cwd=repo_root).returncode


def make_palindrome(buggy: bool) -> FunctionRef:
    """A ``FunctionRef`` for the buggy or the correct palindrome under ``strings/palindrome.py``."""
    source = BUGGY_PALINDROME if buggy else CORRECT_PALINDROME
    return function_ref(source=source, path="strings/palindrome.py", qualname="is_palindrome", lineno=1)


def make_config(hypothesis_examples: int = 200, model_inputs: int = 3) -> CounterexampleConfig:
    """A small ``CounterexampleConfig`` so the sandbox runs stay fast."""
    return CounterexampleConfig(hypothesis_examples=hypothesis_examples, model_inputs=model_inputs)


def prompt_of(client: FakeClient) -> str:
    """The concatenated prompt of the single model call the client recorded."""
    assert len(client.calls) == 1
    return "\n".join(message.content for message in client.calls[0].messages if isinstance(message, Message))


def test_correct_function_returns_none(tmp_path: Path) -> None:
    write_module(tmp_path, "strings/palindrome.py", CORRECT_PALINDROME)
    function = make_palindrome(buggy=False)
    spec = Spec(function=function, predicate=PALINDROME_PREDICATE)
    client = FakeClient([json.dumps({"inputs": ["'aba'", "'abc'"]})])

    found = search(
        function,
        spec,
        tmp_path,
        failed_clause=PALINDROME_CLAUSE,
        client=client,
        model="test-model",
        config=make_config(),
    )

    assert found is None
    assert len(client.calls) == 1
    assert client.calls[0].step == "counterexample"
    prompt = prompt_of(client)
    assert CORRECT_PALINDROME in prompt
    assert PALINDROME_PREDICATE in prompt
    assert PALINDROME_CLAUSE in prompt


def test_buggy_function_found_via_hypothesis_without_model_call(tmp_path: Path) -> None:
    write_module(tmp_path, "strings/palindrome.py", BUGGY_PALINDROME)
    function = make_palindrome(buggy=True)
    spec = Spec(function=function, predicate=PALINDROME_PREDICATE)
    client = FakeClient([json.dumps({"inputs": ["'aba'"]})])

    found = search(
        function, spec, tmp_path, failed_clause=None, client=client, model="test-model", config=make_config()
    )

    assert found is not None
    assert client.calls == []
    assert run_test_file(tmp_path, "test_palindrome_breaks.py", found.test) != 0


def test_unparseable_model_proposal_is_skipped(tmp_path: Path) -> None:
    write_module(tmp_path, "greeter.py", GREET_SOURCE)
    function = function_ref(source=GREET_SOURCE, path="greeter.py", qualname="greet", lineno=1)
    spec = Spec(function=function, predicate=GREET_PREDICATE)
    client = FakeClient([json.dumps({"inputs": ["convincing but ((", "'boom'"]})])

    found = search(
        function, spec, tmp_path, failed_clause=None, client=client, model="test-model", config=make_config(20)
    )

    assert found is not None
    assert found.arguments == "'boom'"


def test_model_inputs_truncates_candidates(tmp_path: Path) -> None:
    write_module(tmp_path, "greeter.py", GREET_SOURCE)
    function = function_ref(source=GREET_SOURCE, path="greeter.py", qualname="greet", lineno=1)
    spec = Spec(function=function, predicate=GREET_PREDICATE)
    client = FakeClient([json.dumps({"inputs": ["'aaa'", "'bbb'", "'boom'"]})])

    found = search(
        function,
        spec,
        tmp_path,
        failed_clause=None,
        client=client,
        model="test-model",
        config=make_config(20, model_inputs=2),
    )

    assert found is None


@pytest.mark.parametrize(
    "answer",
    [
        "not json",
        json.dumps({"no_inputs": []}),
        json.dumps({"inputs": "not a list"}),
        json.dumps({"inputs": ["'ok'", 42]}),
    ],
)
def test_malformed_model_answer_raises_value_error(tmp_path: Path, answer: str) -> None:
    write_module(tmp_path, "greeter.py", GREET_SOURCE)
    function = function_ref(source=GREET_SOURCE, path="greeter.py", qualname="greet", lineno=1)
    spec = Spec(function=function, predicate=GREET_PREDICATE)
    client = FakeClient([answer])

    with pytest.raises(ValueError):
        search(function, spec, tmp_path, failed_clause=None, client=client, model="test-model", config=make_config(20))


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

    write_module(tmp_path, "strings/palindrome.py", CORRECT_PALINDROME)
    function = make_palindrome(buggy=False)
    spec = Spec(function=function, predicate=PALINDROME_PREDICATE)

    with pytest.raises(ToolError, match="model API down"):
        search(
            function,
            spec,
            tmp_path,
            failed_clause=None,
            client=BrokenClient(),
            model="test-model",
            config=make_config(50),
        )


def test_missing_repo_root_raises_tool_error(tmp_path: Path) -> None:
    function = make_palindrome(buggy=False)
    spec = Spec(function=function, predicate=PALINDROME_PREDICATE)
    client = FakeClient([json.dumps({"inputs": ["'aba'"]})])

    with pytest.raises(ToolError):
        search(
            function,
            spec,
            tmp_path / "missing",
            failed_clause=None,
            client=client,
            model="test-model",
            config=make_config(),
        )


GET_SOURCE = "def get(xs: list[int], key: int | None) -> int:\n    return xs[0]\n"
GET_PREDICATE = "def spec(xs, result, key=None):\n    return result == (xs[0] if key is None else xs[key])\n"
GET_PRE = "def pre(xs, key):\n    return len(xs) > 20\n"


def test_keyword_candidate_is_evaluated_in_model_phase(tmp_path: Path) -> None:
    # Hypothesis lists hold at most 20 items, so the precondition never holds there and the model phase
    # decides alone; the probe must stay valid Python for the keyword argument.
    write_module(tmp_path, "getter.py", GET_SOURCE)
    function = function_ref(source=GET_SOURCE, path="getter.py", qualname="get", lineno=1)
    spec = Spec(function=function, predicate=GET_PREDICATE, precondition=GET_PRE)
    client = FakeClient([json.dumps({"inputs": ["list(range(25)), key=5"]})])

    found = search(
        function, spec, tmp_path, failed_clause=None, client=client, model="test-model", config=make_config(20)
    )

    assert found is not None
    assert found.arguments == "list(range(25)), key=5"
    assert found.actual == "0"


def test_model_prompt_includes_failed_clause_and_precondition(tmp_path: Path) -> None:
    write_module(tmp_path, "strings/palindrome.py", CORRECT_PALINDROME)
    function = make_palindrome(buggy=False)
    spec = Spec(function=function, predicate=PALINDROME_PREDICATE, precondition=PRE_NONEMPTY)
    client = FakeClient([json.dumps({"inputs": ["'aba'"]})])

    found = search(
        function,
        spec,
        tmp_path,
        failed_clause=PALINDROME_CLAUSE,
        client=client,
        model="test-model",
        config=make_config(50),
    )

    assert found is None
    prompt = prompt_of(client)
    assert CORRECT_PALINDROME in prompt
    assert PALINDROME_PREDICATE in prompt
    assert PRE_NONEMPTY in prompt
    assert PALINDROME_CLAUSE in prompt
