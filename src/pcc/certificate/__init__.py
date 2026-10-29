"""Certificates under proofs/: writing, reading and detecting stale ones (docs/plan.md, section Certificate)."""

from __future__ import annotations

from pcc.certificate.store import from_result, is_stale, load, write

__all__ = ["from_result", "is_stale", "load", "write"]
