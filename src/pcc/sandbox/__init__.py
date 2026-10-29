"""Running untrusted Python (the code under test, specs, mutants) in a subprocess with time and memory limits.

Every step that executes user code goes through here, so an endless loop or a crash never takes the pipeline down.
"""

from __future__ import annotations

from pcc.sandbox.loader import target_loader
from pcc.sandbox.run import RunOutcome, run_python
from pcc.sandbox.strategies import strategies_source

__all__ = ["RunOutcome", "run_python", "strategies_source", "target_loader"]
