"""Certificates under proofs/: writing, reading and detecting stale ones (docs/plan.md, section Certificate)."""

from __future__ import annotations

from pathlib import Path

from pcc.config import Config
from pcc.core import Certificate, FunctionRef, Result


def from_result(result: Result, config: Config) -> Certificate:
    """Build the certificate of a ``proven`` result. The Dafny version comes from ``config.dafny.version``."""
    raise NotImplementedError


def write(certificate: Certificate, proofs_dir: Path) -> Path:
    """Store the certificate (Dafny file plus JSON metadata) and return the path of the Dafny file."""
    raise NotImplementedError


def load(path: Path) -> Certificate:
    raise NotImplementedError


def is_stale(certificate: Certificate, function: FunctionRef) -> bool:
    """True when the function's source changed since the certificate was written."""
    raise NotImplementedError


__all__ = ["from_result", "is_stale", "load", "write"]
