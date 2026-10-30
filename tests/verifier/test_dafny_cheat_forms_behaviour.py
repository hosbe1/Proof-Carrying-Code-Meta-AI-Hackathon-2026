"""What Dafny 4.11 does with each cheat form that the anti-cheat scan must catch.

This documents Dafny's behaviour; ``pcc.verifier.anti_cheat`` must catch every form below.
The fixtures in ``cheats/`` are the inputs its tests use. Exit code 0 means verified; 2 means only warnings, which are
not verified without ``--allow-warnings`` (never passed); 4 means verification errors.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pcc.config import Config

pytestmark = pytest.mark.dafny

CHEATS_DIR = Path(__file__).parent / "cheats"


@pytest.mark.parametrize(
    ("fixture", "exit_code"),
    [
        ("axiom_attribute.dfy", 0),  # assume {:axiom} false; verifies silently
        ("axiom_attribute_lemma.dfy", 0),  # {:axiom} on a body-less lemma verifies silently
        ("axiom_at_lemma.dfy", 0),  # @Axiom on a body-less lemma verifies silently
        ("assume.dfy", 2),  # plain assume: warning only
        ("body_less_lemma.dfy", 2),  # body-less lemma without {:axiom}: warning only
        ("verify_false.dfy", 2),  # {:verify false}: warning only
        ("only.dfy", 2),  # {:only}: warning only
        ("extern.dfy", 2),  # {:extern} on a body-less method: warning only
        ("expect.dfy", 0),  # expect is only checked at run time; verifies silently
        ("decreases_star.dfy", 0),  # decreases *: no termination check; verifies silently
        ("termination_false.dfy", 0),  # {:termination false} on terminating code verifies; only the scan catches it
        ("resource_limit.dfy", 0),  # resource limit overrides verify; only the scan catches them
        ("selective_checking.dfy", 0),  # {:selective_checking} verifies with exit 0; only the scan catches it
    ],
)
def test_dafny_exit_code_for_each_cheat_form(fixture: str, exit_code: int) -> None:
    proc = subprocess.run(
        [
            "dafny",
            "verify",
            "--resource-limit",
            str(Config().dafny.resource_limit),
            "--verification-time-limit",
            "0",
            str(CHEATS_DIR / fixture),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == exit_code, proc.stdout + proc.stderr
