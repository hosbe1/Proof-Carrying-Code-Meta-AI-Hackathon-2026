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


def _cell_name(cell: str) -> str:
    """The display name of a table cell, unwrapping a markdown link."""
    text = cell.strip()
    if text.startswith("[") and "](" in text:
        return text[1 : text.index("]")].strip()
    return text


def _row_for(name: str) -> str:
    """The single table row whose first cell names ``name``."""
    matches = [row for row in _table_rows() if _cell_name(row.split("|")[1]).lower() == name.lower()]
    assert len(matches) == 1, f"THIRD_PARTY.md must have exactly one table row for {name!r}"
    return matches[0]


def test_source_cells_link_a_direct_source_url() -> None:
    for name in (*LICENSED_ENTRIES, *UNLICENSED_ENTRIES, *BASE_MODEL_ENTRIES):
        source_cell = _row_for(name).split("|")[1]
        assert "http" in source_cell and "](" in source_cell, (
            f"the Source cell for {name!r} must contain a direct source URL"
        )


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


def test_file_contains_no_placeholder_or_todo_claims() -> None:
    text = THIRD_PARTY.read_text(encoding="utf-8").lower()
    for placeholder in (
        "to be confirmed",
        "todo",
        "copyright as stated",
        "as attributed upstream",
        "base-model holders",
        "vericoding contributors",
        "dafnybench contributors",
    ):
        assert placeholder not in text, f"THIRD_PARTY.md must not contain placeholder {placeholder!r}"


def test_vericoding_row_uses_correct_source_and_holder() -> None:
    row = _row_for("Vericoding")
    assert "Beneficial-AI-Foundation/vericoding-benchmark/blob/main/LICENSE" in row, (
        "the Vericoding row must link the vericoding-benchmark LICENSE file"
    )
    assert "Beneficial-AI-Foundation/vericoding-benchmark" in row, (
        "the Vericoding row must link the vericoding-benchmark source"
    )
    assert "Copyright (c) 2025 Beneficial AI Foundation" in row, (
        "the Vericoding row must name the upstream copyright line"
    )
    assert "github.com/vericoding/vericoding" not in THIRD_PARTY.read_text(encoding="utf-8"), (
        "THIRD_PARTY.md must not link the wrong vericoding/vericoding repository"
    )


def test_dafnybench_row_uses_correct_source_and_states_unfilled_notice() -> None:
    row = _row_for("DafnyBench")
    assert "sun-wendy/DafnyBench/blob/main/LICENSE" in row, (
        "the DafnyBench row must link the sun-wendy/DafnyBench LICENSE file"
    )
    assert "unfilled" in row.lower(), (
        "the DafnyBench row must state that upstream LICENSE holds only the unfilled appendix"
    )
    assert "github.com/DafnyBench/DafnyBench" not in THIRD_PARTY.read_text(encoding="utf-8"), (
        "THIRD_PARTY.md must not link the wrong DafnyBench/DafnyBench repository"
    )


def test_mbpp_dfy_row_links_source_license_file() -> None:
    row = _row_for("MBPP-DFY")
    assert "Mondego/vRAG/blob/master/LICENSE" in row, "the MBPP-DFY row must link the vRAG source LICENSE file"
    assert "Mondego/vRAG/tree/master/MBPP-DFY-50" in row, (
        "the MBPP-DFY row must link the vRAG MBPP-DFY source directory"
    )
    assert "gnu.org/licenses/gpl-3.0.html" not in THIRD_PARTY.read_text(encoding="utf-8"), (
        "THIRD_PARTY.md must not use the generic GPL text in place of the source license file"
    )


def test_mbpp_row_cites_upstream_declaration_and_legal_text() -> None:
    row = _row_for("MBPP")
    assert "google-research/google-research/blob/master/README.md" in row, (
        "the MBPP row must cite the upstream README declaring datasets CC BY 4.0"
    )
    assert "creativecommons.org/licenses/by/4.0/legalcode" in row, (
        "the MBPP row must link the canonical CC BY 4.0 legal text"
    )
    assert "Austin" in row and "2021" in row, "the MBPP row must carry the Austin et al., 2021 attribution"


def test_spark_wonder_rows_cite_model_api_terms() -> None:
    for name in ("Muse Spark", "Muse Wonder"):
        row = _row_for(name)
        assert "dev.meta.ai/legal/terms-of-service" in row, f"the {name} row must cite the Model API Terms of Service"
        assert "dev.meta.ai/legal/acceptable-use-policy" in row, f"the {name} row must cite the Acceptable Use Policy"
    assert "meta.ai/legal/commercial-terms" not in THIRD_PARTY.read_text(encoding="utf-8"), (
        "THIRD_PARTY.md must not cite the unverified commercial-terms URL"
    )


def test_glimmer_row_links_base_model_license_and_names_model() -> None:
    row = _row_for("Muse Glimmer")
    assert "huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/LICENSE" in row, (
        "the Glimmer row must link the Muse-Glimmer-30B source LICENSE file"
    )
    assert "huggingface.co/meta-models/Muse-Glimmer-30B" in row, (
        "the Glimmer row must link the Muse-Glimmer-30B base model"
    )
    assert "Meta Superintelligence Lab" in row, (
        "the Glimmer row must attribute the author named on the official model card"
    )
    assert "unfilled" in row.lower(), (
        "the Glimmer row must state that upstream LICENSE holds only the unfilled appendix"
    )
    assert "apache.org/licenses" not in THIRD_PARTY.read_text(encoding="utf-8").lower(), (
        "THIRD_PARTY.md must not use the generic Apache text in place of a source license file"
    )


def test_humanevalfix_row_links_dataset_card_and_octopack_license() -> None:
    row = _row_for("HumanEvalFix")
    assert "huggingface.co/datasets/bigcode/humanevalpack" in row, (
        "the HumanEvalFix row must link the HumanEvalPack dataset source"
    )
    assert "humanevalpack/blob/main/README.md" in row, "the HumanEvalFix row must cite the dataset card declaring MIT"
    assert "bigcode-project/octopack/blob/main/LICENSE" in row, (
        "the HumanEvalFix row must link the Octopack code LICENSE file"
    )
    assert "Muennighoff" in row, "the HumanEvalFix row must name the Octopack copyright holder"
    assert "bigcode-evaluation-harness" not in THIRD_PARTY.read_text(encoding="utf-8"), (
        "THIRD_PARTY.md must not present the evaluation harness as the dataset source"
    )


def test_thealgorithms_notice_names_full_copyright_line() -> None:
    row = _row_for("TheAlgorithms/Python")
    assert "2016-2022" in row, "the demo row must name the full upstream copyright range"
    assert "contributors" in row.lower(), "the demo row must name the contributors of the notice"


def test_readme_links_third_party() -> None:
    text = README.read_text(encoding="utf-8")
    assert "[THIRD_PARTY.md](THIRD_PARTY.md)" in text, "the README must link THIRD_PARTY.md"
