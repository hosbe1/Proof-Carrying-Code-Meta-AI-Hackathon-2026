"""Command line entry point ``pcc-check``, also used by the GitHub Action."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pcc-check", description="Proof-carrying code for Python functions.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="check functions and write certificates")
    run.add_argument("paths", nargs="*", help="files to check (default: functions changed against --base)")
    run.add_argument("--base", default="origin/main", help="git ref to diff against")
    run.add_argument("--proofs-dir", default="proofs")
    sub.add_parser("version", help="print the version")
    args = parser.parse_args(argv)
    if args.command == "version":
        from importlib.metadata import version

        print(version("pcc"))
        return 0
    print("pcc-check run: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
