"""The search agent of step 1 (issue #32): a JSON action loop with blind tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcc.context.search import STEP, search_further
from pcc.core import Context, SourceKind
from pcc.core.errors import ToolError
from pcc.models import Completion, FakeClient, Message, ModelClient
from tests.context.helpers import FUNCTION_SOURCE, action_answer, all_prompt_text, make_context


def make_repo(tmp_path: Path) -> Path:
    package = tmp_path / "src" / "shop.py"
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_text('"""Shop helpers."""\n\n' + FUNCTION_SOURCE)
    caller = tmp_path / "src" / "cart.py"
    caller.write_text("from shop import maximum\n\ntotal = maximum([1, 2])\n")
    return tmp_path


def test_two_grep_turns_then_done_adds_one_call_site(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient(
        [
            action_answer("grep", "maximum"),
            action_answer("grep", "discount"),
            action_answer(
                "done",
                "",
                [{"kind": "call_site", "ref": "src/cart.py:3", "excerpt": "total = maximum([1, 2])"}],
            ),
        ]
    )
    result = search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")

    assert len(result.evidence) == 1
    item = result.evidence[0]
    assert item.kind is SourceKind.CALL_SITE
    assert item.ref == "src/cart.py:3"
    assert len(client.calls) == 3
    assert all(call.step == STEP for call in client.calls)


def test_prompt_never_contains_the_function_body(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient([action_answer("grep", "maximum"), action_answer("done")])
    search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")

    assert FUNCTION_SOURCE not in all_prompt_text(client)


def test_read_file_on_own_file_masks_the_body(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient([action_answer("read_file", "src/shop.py"), action_answer("done")])
    search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")

    assert len(client.calls) == 2
    second_prompt = "\n".join(message.content for message in client.calls[1].messages)
    assert FUNCTION_SOURCE not in second_prompt
    assert "return max(xs)" not in second_prompt


def test_path_escape_returns_an_error_to_the_model(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient([action_answer("read_file", "../secret"), action_answer("done")])
    result = search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")

    assert result.evidence == ()
    second_prompt = "\n".join(message.content for message in client.calls[1].messages)
    assert "error" in second_prompt.lower()
    assert FUNCTION_SOURCE not in second_prompt


def test_max_turns_without_done_returns_what_was_collected(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient(
        [
            action_answer("grep", "maximum", [{"kind": "call_site", "ref": "a:1", "excerpt": "first"}]),
            action_answer("grep", "discount", [{"kind": "docs", "ref": "b:2", "excerpt": "second"}]),
        ]
    )
    result = search_further(make_context(), repo_root, client=client, model="muse-spark-1.3", max_turns=2)

    assert [item.excerpt for item in result.evidence] == ["first", "second"]
    assert len(client.calls) == 2


def test_evidence_with_the_function_source_is_dropped(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient(
        [action_answer("done", "", [{"kind": "docs", "ref": "src/shop.py:3", "excerpt": FUNCTION_SOURCE}])]
    )
    result = search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")

    assert result.evidence == ()


def test_evidence_with_an_unknown_kind_is_dropped(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient(
        [action_answer("done", "", [{"kind": "rumor", "ref": "x:1", "excerpt": "some gossip"}])]
    )
    result = search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")

    assert result.evidence == ()


def test_blindness_holds_over_tool_results_and_evidence(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient(
        [
            action_answer("read_file", "src/shop.py"),
            action_answer("grep", "maximum"),
            action_answer(
                "done",
                "",
                [{"kind": "call_site", "ref": "src/cart.py:3", "excerpt": "total = maximum([1, 2])"}],
            ),
        ]
    )
    result = search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")

    assert FUNCTION_SOURCE not in all_prompt_text(client)
    for item in result.evidence:
        assert FUNCTION_SOURCE not in item.excerpt


def test_tool_error_passes_through(tmp_path: Path) -> None:
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

    with pytest.raises(ToolError, match="model API down"):
        search_further(make_context(), tmp_path, client=BrokenClient(), model="muse-spark-1.3")


def test_malformed_answer_raises_value_error(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient(["not json"])
    with pytest.raises(ValueError, match="JSON"):
        search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")


def test_unknown_action_raises_value_error(tmp_path: Path) -> None:
    repo_root = make_repo(tmp_path)
    client = FakeClient([json.dumps({"action": "delete", "argument": "x", "evidence": []})])
    with pytest.raises(ValueError, match="action"):
        search_further(make_context(), repo_root, client=client, model="muse-spark-1.3")


def test_existing_evidence_is_kept(tmp_path: Path) -> None:
    from pcc.core import Evidence

    from tests.helpers import function_ref

    repo_root = make_repo(tmp_path)
    function = function_ref(source=FUNCTION_SOURCE, path="src/shop.py", qualname="maximum", lineno=3)
    base = Context(
        function=function,
        evidence=(Evidence(SourceKind.DOCSTRING, "maximum:3", "Return the largest element."),),
    )
    client = FakeClient(
        [action_answer("done", "", [{"kind": "call_site", "ref": "src/cart.py:3", "excerpt": "uses it"}])]
    )
    result = search_further(base, repo_root, client=client, model="muse-spark-1.3")

    assert [item.kind for item in result.evidence] == [SourceKind.DOCSTRING, SourceKind.CALL_SITE]
    assert result.function == function
