"""The shared system message of every training data line.

``SYSTEM_MESSAGE`` is sent verbatim by the rejection sampling tool and the SFT export, and by
the evaluation that sends the same prompt.
"""

from __future__ import annotations

SYSTEM_MESSAGE = (
    "You are a Dafny 4.11 verification assistant. Return only a complete Dafny program. "
    "Preserve the supplied specification and method declaration exactly. "
    "If executable code is supplied, preserve it and add only ghost proof annotations. "
    "Do not use assume, axioms, verification-disabling attributes, or resource-limit overrides. "
    "The program must verify without --allow-warnings."
)

__all__ = ["SYSTEM_MESSAGE"]
