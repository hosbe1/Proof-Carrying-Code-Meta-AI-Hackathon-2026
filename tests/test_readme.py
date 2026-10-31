"""Acceptance tests for the README (issue #51).

Covers the ticket interface and examples: ``README.md`` has every required
section, the model table matches ``docs/plan.md``, the quick start snippet
carries what the demo repository needs, and every relative link resolves to a
file in this repository. The tests read only local files and need no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _headings() -> list[str]:
    return [line.strip() for line in _text().splitlines() if line.startswith("## ")]


def _fences(language: str) -> list[str]:
    return [body for tag, body in _FENCE_RE.findall(_text()) if tag == language]


def test_readme_exists() -> None:
    assert README.is_file(), "README.md must exist at the repository root"


@pytest.mark.parametrize(
    "heading",
    (
        "## What it does",
        "## Quick start",
        "## How it works",
        "## Which Meta model does what",
        "## Results",
        "## Licensing",
        "## Team",
    ),
)
def test_required_section_present(heading: str) -> None:
    assert heading in _headings(), f"README.md must have a {heading!r} section"


def test_what_it_does_has_one_paragraph_and_one_example_comment() -> None:
    body = _text()
    section = body.split("## What it does", 1)[1].split("## ", 1)[0]
    paragraphs = [block.strip() for block in section.strip().split("\n\n") if block.strip()]
    prose = [block for block in paragraphs if not block.startswith("```")]
    assert len(prose) >= 1, "the What-it-does section needs an introductory paragraph"
    assert "<!-- pcc-check -->" in section, "the What-it-does section needs one example pull request comment"


def test_quick_start_snippet_has_what_the_demo_needs() -> None:
    snippets = _fences("yaml")
    assert snippets, "the Quick-start section needs a yaml workflow snippet"
    snippet = "\n".join(snippets)
    assert "pull-requests: write" in snippet, "the snippet must request the comment permission"
    assert "META_API_KEY" in snippet, "the snippet must pass the Meta Model API key secret"
    assert "pcc-check run" in snippet or "uses:" in snippet, (
        "the snippet must run this repository's action (pcc-check run)"
    )


def test_how_it_works_links_the_plan() -> None:
    assert "[docs/plan.md](docs/plan.md)" in _text(), "the How-it-works section must link docs/plan.md"


def test_model_table_matches_the_plan() -> None:
    body = _text()
    section = body.split("## Which Meta model does what", 1)[1].split("## ", 1)[0]
    for model in ("muse-spark-1.3", "muse-wonder-1.0"):
        assert model in section, f"the model table must name {model} exactly as docs/plan.md does"
    assert "Glimmer" in section, "the model table must cover the trained Glimmer proof stage"
    lowered = section.lower()
    assert "intent" in lowered and "spec" in lowered, "Spark must be shown on intent and spec"
    assert "wonder" in lowered and "40" in section, "Wonder must be shown as proof stage 3 for short functions"


def test_results_section_names_the_bug_hunt_report() -> None:
    body = _text()
    section = body.split("## Results", 1)[1].split("## ", 1)[0]
    assert "docs/results/bug-hunt.md" in section, "the Results section must name docs/results/bug-hunt.md"


def test_results_section_has_no_dead_link_to_missing_report() -> None:
    targets = _LINK_RE.findall(_text())
    assert "docs/results/bug-hunt.md" not in targets, (
        "docs/results/bug-hunt.md does not exist yet, so the README must not link it"
    )


def test_licensing_links_third_party() -> None:
    body = _text()
    section = body.split("## Licensing", 1)[1].split("## ", 1)[0]
    assert "[THIRD_PARTY.md](THIRD_PARTY.md)" in section, "the Licensing section must link THIRD_PARTY.md"


def test_every_relative_link_resolves() -> None:
    missing = []
    for target in _LINK_RE.findall(_text()):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (REPO_ROOT / target.split("#")[0]).resolve()
        try:
            path.relative_to(REPO_ROOT.resolve())
        except ValueError:
            missing.append(target)
            continue
        if not path.is_file():
            missing.append(target)
    assert not missing, f"every relative README link must resolve to a file: {missing}"


def test_every_external_link_is_https() -> None:
    for target in _LINK_RE.findall(_text()):
        if "://" not in target:
            continue
        assert target.startswith("https://"), f"external README link must be https: {target}"
