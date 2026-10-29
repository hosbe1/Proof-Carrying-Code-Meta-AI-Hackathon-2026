"""The pull request comment (docs/plan.md, section Pull request comment)."""

from __future__ import annotations

from collections.abc import Sequence

from pcc.core import (
    Certificate,
    CheckKind,
    Intent,
    Result,
    SecondOpinion,
    State,
    Translation,
    TranslationPath,
)

MARKER = "<!-- pcc-check -->"
HEADING = "## Proof-carrying code"
MAX_LENGTH = 65_000
PRE_WARNING_SHARE = 20

_ICONS = {
    State.PROVEN: "✅",
    State.COUNTEREXAMPLE: "❌",
    State.OPEN: "⚪",
    State.NOT_CHECKABLE: "➖",  # noqa: RUF001 - the ticket mandates this icon for not checkable
}

_STATES = {
    State.PROVEN: "proven",
    State.COUNTEREXAMPLE: "counterexample",
    State.OPEN: "open",
    State.NOT_CHECKABLE: "not checkable",
}


def render(results: Sequence[Result], *, stale: Sequence[Certificate] = ()) -> str:
    """Markdown: summary line on top, counterexamples first, then per function state, translation path, intent with
    sources, counterexample as a ready-to-copy test, assumptions, and the Dafny spec collapsed."""
    ordered = sorted(
        results,
        key=lambda result: (
            result.state is not State.COUNTEREXAMPLE,
            result.function.path,
            result.function.lineno,
            result.function.qualname,
        ),
    )
    shortened: set[int] = set()
    sections = [_section(result, shortened=False) for result in ordered]
    note = ""
    if _length(ordered, sections, note, stale) > MAX_LENGTH:
        shortened = {index for index, result in enumerate(ordered) if result.state is State.PROVEN}
        sections = [_section(result, index in shortened) for index, result in enumerate(ordered)]
        if _length(ordered, sections, note, stale) > MAX_LENGTH:
            shortened = {index for index, result in enumerate(ordered) if result.state in (State.PROVEN, State.OPEN)}
            sections = [_section(result, index in shortened) for index, result in enumerate(ordered)]
        note = f"Note: details of {len(shortened)} functions shortened to fit the comment length limit."
    parts = [MARKER, HEADING, "", _summary(ordered)]
    if note:
        parts += ["", note]
    if sections:
        parts += ["", *blank_separated(sections)]
    if stale:
        parts += [
            "",
            "## Stale certificates",
            *[f"- `{certificate.function.path}::{certificate.function.qualname}`" for certificate in stale],
        ]
    parts += ["", _cost_line(ordered)]
    return "\n".join(parts)


def blank_separated(sections: list[str]) -> list[str]:
    """The sections with one blank line between them."""
    joined: list[str] = []
    for section in sections:
        if joined:
            joined.append("")
        joined.append(section)
    return joined


def _length(results: Sequence[Result], sections: list[str], note: str, stale: Sequence[Certificate]) -> int:
    """The length of the comment the parts above would produce."""
    parts = [MARKER, HEADING, "", _summary(results)]
    if note:
        parts += ["", note]
    if sections:
        parts += ["", *blank_separated(sections)]
    if stale:
        parts += [
            "",
            "## Stale certificates",
            *[f"- `{certificate.function.path}::{certificate.function.qualname}`" for certificate in stale],
        ]
    parts += ["", _cost_line(results)]
    return len("\n".join(parts))


def _summary(results: Sequence[Result]) -> str:
    """Counts in state order, leaving out zero counts; ``No functions checked.`` when there are none."""
    counts = [(state, sum(1 for result in results if result.state is state)) for state in State]
    if all(count == 0 for _, count in counts):
        return "No functions checked."
    return ", ".join(f"{count} {_plural(state, count)}" for state, count in counts if count)


def _plural(state: State, count: int) -> str:
    """The state word in the summary line; only ``counterexample`` changes in the plural."""
    if state is State.COUNTEREXAMPLE and count != 1:
        return "counterexamples"
    return _STATES[state]


def _section(result: Result, shortened: bool) -> str:
    """One function: the heading line, the reason when shortened, the full detail otherwise."""
    function = result.function
    lines = [f"### {_ICONS[result.state]} `{function.path}::{function.qualname}` — {_STATES[result.state]}"]
    if shortened:
        if result.reason:
            lines.append(f"Reason: {result.reason}")
        return "\n".join(lines)
    lines += _detail(result)
    return "\n".join(lines)


def _detail(result: Result) -> list[str]:
    """Everything below the heading line: pre, reason, translation path, intent, counterexample, assumptions, spec."""
    lines = _pre_lines(result)
    if result.reason:
        lines.append(f"Reason: {result.reason}")
    if result.translation is not None:
        lines.append(_translation_line(result.translation))
    if result.intent is not None:
        lines += _intent_lines(result.intent)
    if result.second_opinion is not None:
        lines.append(_second_opinion_line(result.second_opinion))
    if result.state is State.COUNTEREXAMPLE and result.counterexample is not None:
        lines += _counterexample_lines(result)
    if result.assumptions:
        lines += _assumption_lines(result)
    if result.translation is not None:
        lines += [
            "<details><summary>Dafny spec</summary>",
            "",
            "```dafny",
            result.translation.dafny_spec,
            "```",
            "",
            "</details>",
        ]
    return lines


def _pre_lines(result: Result) -> list[str]:
    """The ``pre`` line with the share of inputs it accepts, plus a warning below 20 %."""
    if result.spec is None or not result.spec.precondition:
        return []
    share = _pre_share(result)
    line = f"pre: `{result.spec.precondition}`"
    if share is None:
        return [line]
    lines = [f"{line} (accepts {share} % of inputs)"]
    if share < PRE_WARNING_SHARE:
        lines.append(f"⚠ pre accepts only {share} % of generated inputs")
    return lines


def _pre_share(result: Result) -> int | None:
    """The rounded percentage of generated inputs the precondition accepts, if it was recorded."""
    if result.spec is None:
        return None
    for record in result.spec.checks:
        if record.kind is CheckKind.HYPOTHESIS_ORIGINAL and record.inputs_tried is not None and record.inputs_total:
            return round(record.inputs_tried * 100 / record.inputs_total)
    return None


def _translation_line(translation: Translation) -> str:
    """Whether the translation came from the transpiler or from AI."""
    if translation.function_path is TranslationPath.TRANSPILER:
        return "translated by the transpiler"
    return "translated by AI, tested, not guaranteed"


def _intent_lines(intent: Intent) -> list[str]:
    """The intent summary, each claim with its source refs, and the contradictions."""
    lines = [f"Intent: {intent.summary}"]
    for claim_text in intent.claims:
        sources = ", ".join(f"{source.kind.value} {source.ref}" for source in claim_text.sources)
        lines.append(f"- {claim_text.text} [{sources}]" if sources else f"- {claim_text.text}")
    if intent.contradictions:
        lines.append("Contradictions:")
        for contradiction in intent.contradictions:
            resolved = f"resolved by: {contradiction.resolved_by.value}" if contradiction.resolved_by else "unresolved"
            lines.append(f"- {contradiction.description} ({resolved})")
    return lines


def _second_opinion_line(opinion: SecondOpinion) -> str:
    """A one-line second-opinion note."""
    if opinion.note:
        return f"Second opinion: {opinion.note}"
    return f"Second opinion: {'agrees.' if opinion.agrees else 'disagrees.'}"


def _counterexample_lines(result: Result) -> list[str]:
    """The call, the expected and actual behavior, the ready-to-run test and the suggested fix."""
    counterexample = result.counterexample
    assert counterexample is not None
    lines = [
        f"Call: `{result.function.qualname}({counterexample.arguments})`",
        f"Expected: {counterexample.expected}",
        f"Actual: {counterexample.actual}",
        "```python",
        counterexample.test,
        "```",
    ]
    if counterexample.suggested_fix:
        lines += [
            "Possible fix (a mutant that satisfies the spec):",
            "```diff",
            counterexample.suggested_fix,
            "```",
        ]
    return lines


def _assumption_lines(result: Result) -> list[str]:
    """The assumptions the result rests on."""
    count = len(result.assumptions)
    noun = "assumption" if count == 1 else "assumptions"
    header = f"proven under {count} {noun}:" if result.state is State.PROVEN else f"with {count} {noun}:"
    lines = [header]
    for assumption in result.assumptions:
        lines.append(f"- **{assumption.subject}** ({assumption.kind.value}): {assumption.statement}")
    return lines


def _cost_line(results: Sequence[Result]) -> str:
    """Total cost and time over all ``Usage`` records."""
    cost = sum(entry.cost_usd for result in results for entry in result.usage)
    seconds = round(sum(entry.seconds for result in results for entry in result.usage))
    if 0 < cost < 0.01:
        shown = "<$0.01"
    elif cost == 0:
        shown = "$0.00"
    else:
        shown = f"${cost:.2f}"
    shown_time = f"{seconds} s" if seconds < 60 else f"{seconds // 60} min {seconds % 60} s"
    return f"Cost {shown} · {shown_time}"


__all__ = ["render"]
