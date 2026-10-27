"""Evaluation runs on fixed test splits (docs/plan.md, "Evaluation"): proof rate, bugs found, false alarms, share per
Python tier, share translated by the transpiler, cost and time per function."""

from __future__ import annotations

from pathlib import Path


def run(dataset: str, split: str, out_dir: Path) -> Path:
    """Run the pipeline on one dataset split and write a metrics report; return its path."""
    raise NotImplementedError


__all__ = ["run"]
