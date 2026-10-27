from __future__ import annotations

from importlib.metadata import version

import pytest

from pcc.cli import main


def test_version_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out == f"{version('pcc')}\n"


def test_run_is_not_implemented_yet(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run"]) == 2
    assert "not implemented yet" in capsys.readouterr().err
