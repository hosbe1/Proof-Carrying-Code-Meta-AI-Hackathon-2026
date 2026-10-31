"""The shared prompt pieces of the training path (docs/plan.md, Training).

The canonical objects live in ``pcc.evaluation.prompts`` (the package never imports
``training``, see docs/architecture.md); this module re-exports them unchanged so training
imports keep working and every prompt compares byte-identical across evaluation, training
and the SFT export guard.
"""

from __future__ import annotations

from pcc.evaluation.prompts import SYSTEM_MESSAGE, build_prompt

__all__ = ["SYSTEM_MESSAGE", "build_prompt"]
