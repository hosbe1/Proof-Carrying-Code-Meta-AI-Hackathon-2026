"""The only way to reach a language model.

Steps receive a ``ModelClient`` as an argument and never build one themselves, so tests run against
``FakeClient`` and CI never needs API keys. Real clients (Meta Model API, hosted Glimmer) live in this package too.
"""

from __future__ import annotations

from pcc.models.client import Completion, Message, ModelClient, Role
from pcc.models.fake import Call, FakeClient
from pcc.models.router import Router, available_stages

__all__ = [
    "Call",
    "Completion",
    "FakeClient",
    "Message",
    "ModelClient",
    "Role",
    "Router",
    "available_stages",
]
