"""The only way to reach a language model.

Steps receive a ``ModelClient`` as an argument and never build one themselves, so tests run against
``FakeClient`` and CI never needs API keys. Real clients (Meta Model API, hosted Glimmer) live in this package too.
"""

from __future__ import annotations

from pcc.models.budget import BudgetedClient, BudgetExceeded
from pcc.models.client import Completion, Message, ModelClient, Role
from pcc.models.fake import Call, FakeClient
from pcc.models.openai_compat import OpenAICompatClient
from pcc.models.router import ModelUnavailable, Router, available_stages, build_client

__all__ = [
    "BudgetExceeded",
    "BudgetedClient",
    "Call",
    "Completion",
    "FakeClient",
    "Message",
    "ModelClient",
    "ModelUnavailable",
    "OpenAICompatClient",
    "Role",
    "Router",
    "available_stages",
    "build_client",
]
