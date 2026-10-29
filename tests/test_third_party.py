"""Acceptance tests for the third-party attribution file.

Covers the ticket acceptance criteria: every ``THIRD_PARTY.md`` table row links
the license file of its source (or states that none exists), and the README
links ``THIRD_PARTY.md``. The tests read only local files and need no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "THIRD_PARTY.md"
README = REPO_ROOT / "README.md"

# Entries whose source is published under a license file we can link.
LICENSED_ENTRIES = (
    "HumanEvalFix",
    "HumanEval",
    "MBPP",
    "Vericoding",
    "DafnyBench",
    "MBPP-DFY",
    "TheAlgorithms/Python",
)

# Entries with no license file upstream: the row must link the source or the
# applicable terms and say so.
UNLICENSED_ENTRIES = (
    "BugsInPy",
    "Muse Spark",
    "Muse Wonder",
)

# Entries whose base model is published under a license file, while the
# artifact we use (the trained adapter) is released separately.
BASE_MODEL_ENTRIES = ("Muse Glimmer",)

# Entries whose license text requires a copyright notice or attribution.
NOTICE_ENTRIES = (
    "HumanEvalFix",
    "HumanEval",
    "MBPP",
    "Vericoding",
    "DafnyBench",
    "TheAlgorithms/Python",
    "Muse Glimmer",
)


def _table_rows() -> list[str]:
    """The markdown table rows of THIRD_PARTY.md."""
    return [line for line in THIRD_PARTY.read_text(encoding="utf-8").splitlines() if line.startswith("|")]


def _row_for(name: str) -> str:
    """The single table row whose first cell names ``name``."""
    matches = [row for row in _table_rows() if row.split("|")[1].strip().lower() == name.lower()]
    assert len(matches) == 1, f"THIRD_PARTY.md must have exactly one table row for {name!r}"
    return matches[0]


def test_third_party_file_exists() -> None:
    assert THIRD_PARTY.is_file(), "THIRD_PARTY.md must exist at the repository root"


def test_table_header_has_required_columns() -> None:
    header = _table_rows()[0].lower()
    for column in ("source", "license", "use", "commit"):
        assert column in header, f"table header must name a {column!r} column"


def test_table_covers_every_dataset_demo_source_and_model() -> None:
    for name in (*LICENSED_ENTRIES, *UNLICENSED_ENTRIES, *BASE_MODEL_ENTRIES):
        _row_for(name)


@pytest.mark.parametrize("name", LICENSED_ENTRIES)
def test_row_links_the_license_file_of_its_source(name: str) -> None:
    row = _row_for(name)
    assert "http" in row, f"row for {name!r} must link its source"
    assert "[" in row and "](" in row, f"row for {name!r} must contain a markdown link"
    assert "license" in row.lower(), f"row for {name!r} must link or name the license text"


@pytest.mark.parametrize("name", UNLICENSED_ENTRIES)
def test_row_without_license_file_links_source_and_says_so(name: str) -> None:
    row = _row_for(name)
    assert "http" in row, f"row for {name!r} must link its source or terms"
    lowered = row.lower()
    assert "no license file" in lowered or "proprietary" in lowered, (
        f"row for {name!r} must state that no license file exists upstream"
    )


@pytest.mark.parametrize("name", BASE_MODEL_ENTRIES)
def test_row_links_the_base_model_license(name: str) -> None:
    row = _row_for(name)
    lowered = row.lower()
    assert "http" in row, f"row for {name!r} must link the base-model license text"
    assert "apache" in lowered, f"row for {name!r} must name the Apache-2.0 base-model license"
    assert "base" in lowered, f"row for {name!r} must say it is the base-model license"


@pytest.mark.parametrize("name", NOTICE_ENTRIES)
def test_notice_licenses_name_their_copyright_line(name: str) -> None:
    row = _row_for(name).lower()
    assert "copyright" in row or "attribution" in row or "\u00a9" in row, (
        f"row for {name!r} needs a notice, so it must name its copyright line or attribution"
    )


def test_vericoding_row_excludes_verifiedcogen_with_reason() -> None:
    row = _row_for("Vericoding").lower()
    assert "verifiedcogen" in row, "the Vericoding row must name the excluded VerifiedCogen part"
    assert "exclud" in row or "not used" in row, "the Vericoding row must state the exclusion"
    assert "unlicensed" in row or "license" in row, "the Vericoding row must give the licensing reason"


def test_demo_row_names_the_copied_files_with_notice() -> None:
    row = _row_for("TheAlgorithms/Python")
    assert "palindrome.py" in row and "z_function.py" in row, "the demo row must name both copied files"
    assert "notice" in row.lower(), "the demo row must state that the files travel with the notice"


def test_bugsinpy_row_is_referenced_by_id_only() -> None:
    row = _row_for("BugsInPy").lower()
    assert "id" in row, "the BugsInPy row must say it is referenced by ID only"
    assert "commit" in row, "the BugsInPy row must say whether anything is committed"


def test_mbpp_dfy_row_is_local_only() -> None:
    row = _row_for("MBPP-DFY").lower()
    assert "local" in row, "the MBPP-DFY row must say it is used locally only"
    assert "commit" in row, "the MBPP-DFY row must say whether anything is committed"


def test_readme_links_third_party() -> None:
    text = README.read_text(encoding="utf-8")
    assert "[THIRD_PARTY.md](THIRD_PARTY.md)" in text, "the README must link THIRD_PARTY.md"
