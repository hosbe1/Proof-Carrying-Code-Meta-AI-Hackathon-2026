"""Test helpers for the context search agent tests (issue #32).

The tools never return the function body: lines of the function under check are replaced with
``OMITTED`` in every result, and paths are confined to ``repo_root``.
"""

from __future__ import annotations

import json

from pcc.core import Context
from pcc.models import FakeClient, Message
from tests.helpers import function_ref

FUNCTION_SOURCE = "def maximum(xs: list[int]) -> int:\n    return max(xs)\n"


def all_prompt_text(client: FakeClient) -> str:
    """Every prompt ever sent through ``client``, joined for blindness checks."""
    chunks: list[str] = []
    for call in client.calls:
        for message in call.messages:
            if isinstance(message, Message):
                chunks.append(message.content)
    return "\n".join(chunks)


def action_answer(action: str, argument: str = "", evidence: list[dict[str, str]] | None = None) -> str:
    """One scripted agent turn as JSON."""
    return json.dumps({"action": action, "argument": argument, "evidence": evidence or []})


def make_context(source: str = FUNCTION_SOURCE) -> Context:
    function = function_ref(source=source, path="src/shop.py", qualname="maximum", lineno=3)
    return Context(function=function, evidence=())
