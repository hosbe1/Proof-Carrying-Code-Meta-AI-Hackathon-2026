"""Entry point for ``python -m pcc.evaluation``: dataset harnesses (docs/plan.md, "Evaluation")."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pcc.core import ToolError
from pcc.evaluation.humanevalfix import run_harness


def build_parser() -> argparse.ArgumentParser:
    """The command line: one subcommand per harness, ``humanevalfix`` first."""
    parser = argparse.ArgumentParser(prog="python -m pcc.evaluation", description="Run an evaluation harness.")
    sub = parser.add_subparsers(dest="harness", required=True)
    fix = sub.add_parser("humanevalfix", help="the whole pipeline on each HumanEvalFix task, fixed and buggy")
    fix.add_argument("--split", required=True, choices=("test", "train"))
    fix.add_argument("--out", required=True, help="output directory, e.g. runs/<name>")
    fix.add_argument("--max-cost-usd", required=True, type=float)
    fix.add_argument("--stages", default=None, help="keep only the prover stage with this name")
    fix.add_argument("--contributor", action="store_true", help="run Spark models in contributor mode")
    fix.add_argument("--spec-model", default=None, help="model id for ModelsConfig.spec")
    fix.add_argument("--spec-only", action="store_true", help="run steps 1 to 3 only, no translation or proof")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the harness named in ``argv``; return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        run_harness(
            split=args.split,
            out_dir=Path(args.out),
            max_cost_usd=args.max_cost_usd,
            stages=args.stages,
            contributor=args.contributor,
            spec_model=args.spec_model,
            spec_only=args.spec_only,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
