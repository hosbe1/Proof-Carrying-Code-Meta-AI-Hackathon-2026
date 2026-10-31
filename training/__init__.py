"""Training data and the graded reward for the proof step (docs/plan.md, Training).

This package lives outside ``src/pcc``: it imports ``pcc`` (the verifier as the reward) but
the package never imports it.
"""

from __future__ import annotations

from training.prompts import SYSTEM_MESSAGE
from training.reward import reward

__all__ = ["SYSTEM_MESSAGE", "reward"]
