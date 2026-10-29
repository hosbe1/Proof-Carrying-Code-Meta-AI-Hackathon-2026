"""Running untrusted Python (the code under test, specs, mutants) in a subprocess with time and memory limits.

Every step that executes user code goes through here, so an endless loop or a crash never takes the pipeline down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pcc.sandbox.loader import target_loader
from pcc.sandbox.strategies import strategies_source


@dataclass(frozen=True, slots=True)
class RunOutcome:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float


def run_python(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
    """Run ``script`` with the project's Python in ``cwd`` under the given limits.

    Contract: the child process gets a minimal environment, built from an explicit allowlist of variables (for example
    ``PATH``, ``HOME``, ``LANG`` and ``PYTHONHASHSEED``). Nothing else from the parent environment is passed on; API
    keys and other secrets never reach the untrusted code.
    """
    raise NotImplementedError


__all__ = ["RunOutcome", "run_python", "strategies_source", "target_loader"]
