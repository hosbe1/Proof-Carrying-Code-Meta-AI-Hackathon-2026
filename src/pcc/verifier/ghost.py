"""The ghost-only check on proof attempts."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pcc.verifier.anti_cheat import (
    _blanked,
    _brace_violations,
    _literal_spans,
    _match_brace,
    _match_paren,
    _normalize,
)

_CLAUSE_STARTS = ("invariant", "decreases", "modifies")
_CLAUSE_KEYWORDS = frozenset({"invariant", "decreases", "modifies", "requires", "ensures", "reads"})

_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _word_pattern(word: str) -> re.Pattern[str]:
    pattern = _WORD_RE_CACHE.get(word)
    if pattern is None:
        pattern = re.compile(r"(?<![A-Za-z0-9_'])" + re.escape(word) + r"(?![A-Za-z0-9_'])")
        _WORD_RE_CACHE[word] = pattern
    return pattern


def _read_word(text: str, pos: int) -> str:
    end = pos
    while end < len(text) and (text[end].isalnum() or text[end] in "_'"):
        end += 1
    return text[pos:end]


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _clause_spans(masked: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for keyword in _CLAUSE_STARTS:
        for match in _word_pattern(keyword).finditer(masked):
            end = _clause_end(masked, match.end())
            if end > match.start():
                spans.append((match.start(), end))
    return spans


def _clause_end(masked: str, pos: int) -> int:
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    i = pos
    n = len(masked)
    while i < n:
        char = masked[i]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            if depth_paren > 0:
                depth_paren -= 1
            elif depth_brack == 0 and depth_brace == 0:
                return i
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            if depth_brack > 0:
                depth_brack -= 1
            elif depth_paren == 0 and depth_brace == 0:
                return i
        elif char == "{":
            if depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
                return i
            depth_brace += 1
        elif char == "}":
            if depth_brace > 0:
                depth_brace -= 1
            elif depth_paren == 0 and depth_brack == 0:
                return i
        elif char == ";" and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            return i
        elif char.isalpha() and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            word = _read_word(masked, i)
            if word in _CLAUSE_KEYWORDS:
                return i
            i += len(word)
            continue
        i += 1
    return n


def _assert_spans(masked: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _word_pattern("assert").finditer(masked):
        span = _assert_end(masked, match.start(), match.end())
        if span is not None:
            spans.append(span)
    return spans


def _assert_end(masked: str, start: int, pos: int) -> tuple[int, int] | None:
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    i = pos
    n = len(masked)
    while i < n:
        char = masked[i]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            if depth_paren > 0:
                depth_paren -= 1
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            if depth_brack > 0:
                depth_brack -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            if depth_brace > 0:
                depth_brace -= 1
            elif depth_paren == 0 and depth_brack == 0:
                return (start, i)
        elif char == ";" and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            return (start, i + 1)
        elif char.isalpha() and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            word = _read_word(masked, i)
            if word == "by":
                return _assert_by_end(masked, start, i + len(word))
            i += len(word)
            continue
        i += 1
    return (start, n)


def _assert_by_end(masked: str, start: int, pos: int) -> tuple[int, int]:
    n = len(masked)
    j = _skip_ws(masked, pos)
    if j < n and masked[j] == "{":
        closing = _match_brace(masked, j)
        if closing is None:
            return (start, n)
        after = _skip_ws(masked, closing + 1)
        if after < n and masked[after] == ";":
            return (start, after + 1)
        return (start, closing + 1)
    depth_paren = 0
    depth_brack = 0
    i = j
    while i < n:
        char = masked[i]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            if depth_paren > 0:
                depth_paren -= 1
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            if depth_brack > 0:
                depth_brack -= 1
        elif char == ";" and depth_paren == 0 and depth_brack == 0:
            return (start, i + 1)
        elif char == "}" and depth_paren == 0 and depth_brack == 0:
            return (start, i)
        i += 1
    return (start, n)


def _calc_spans(masked: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _word_pattern("calc").finditer(masked):
        i = match.end()
        n = len(masked)
        while i < n and masked[i] != "{" and masked[i] != ";" and masked[i] != "}":
            i += 1
        if i < n and masked[i] == "{":
            closing = _match_brace(masked, i)
            if closing is None:
                spans.append((match.start(), n))
            else:
                spans.append((match.start(), closing + 1))
    return spans


def _reveal_spans(masked: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _word_pattern("reveal").finditer(masked):
        end = _statement_end(masked, match.end())
        spans.append((match.start(), end))
    return spans


def _ghost_var_spans(masked: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _word_pattern("ghost").finditer(masked):
        after = _skip_ws(masked, match.end())
        var_word = _read_word(masked, after)
        if var_word != "var":
            continue
        end = _statement_end(masked, after + len(var_word))
        spans.append((match.start(), end))
    return spans


def _statement_end(masked: str, pos: int) -> int:
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    i = pos
    n = len(masked)
    while i < n:
        char = masked[i]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            if depth_paren > 0:
                depth_paren -= 1
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            if depth_brack > 0:
                depth_brack -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            if depth_brace > 0:
                depth_brace -= 1
            elif depth_paren == 0 and depth_brack == 0:
                return i
        elif char == ";" and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            return i + 1
        i += 1
    return n


def _lemma_spans(masked: str, lemma_names: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    names = list(dict.fromkeys(lemma_names))
    for name in names:
        if name == "":
            continue
        for match in _word_pattern(name).finditer(masked):
            span = _lemma_call_span(masked, match.start(), match.end())
            if span is not None:
                spans.append(span)
    return spans


def _lemma_call_span(masked: str, start: int, end: int) -> tuple[int, int] | None:
    back = start - 1
    while back >= 0 and masked[back].isspace():
        back -= 1
    if back >= 0:
        if masked[back] in ".:":
            return None
        if masked[back] not in ";{}":
            return None
    n = len(masked)
    after = _skip_ws(masked, end)
    if after >= n or masked[after] != "(":
        return None
    closing = _match_paren(masked, after)
    if closing is None:
        return None
    after_call = _skip_ws(masked, closing + 1)
    if after_call >= n or masked[after_call] != ";":
        return None
    return (start, after_call + 1)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def executable_part(dafny_method: str, *, lemma_names: Sequence[str]) -> str:
    """Return ``dafny_method`` with comments and ghost code removed and whitespace collapsed.

    Removed ghost forms are ``invariant``/``decreases``/``modifies`` clauses,
    ``assert`` statements (including a ``by`` expression or ``by { ... }`` block),
    ``calc { ... }`` blocks, ``reveal ...;`` statements, ``ghost var ...;``
    declarations and bare calls ``Name(args);`` whose ``Name`` is in
    ``lemma_names``. Everything else (``requires``, ``ensures``, ``reads``,
    ``ghost const`` and friends) is kept, so unhandled additions compare
    unequal. String and character literals are kept verbatim; only comments
    are removed. The function never raises.
    """
    comments, strings = _literal_spans(dafny_method)
    masked = _blanked(dafny_method, comments + strings)
    spans: list[tuple[int, int]] = []
    spans.extend(_clause_spans(masked))
    spans.extend(_assert_spans(masked))
    spans.extend(_calc_spans(masked))
    spans.extend(_reveal_spans(masked))
    spans.extend(_ghost_var_spans(masked))
    spans.extend(_lemma_spans(masked, lemma_names))
    chars = list(dafny_method)
    for start, end in _merge_spans(spans):
        for index in range(start, min(end, len(chars))):
            if chars[index] != "\n":
                chars[index] = " "
    return _normalize("".join(chars))


def same_executable(original: str, proven: str, *, lemma_names: Sequence[str]) -> bool:
    """Return True when ``proven`` differs from ``original`` only in ghost code.

    Ghost code is what :func:`executable_part` removes. Either side with
    unbalanced braces returns False.
    """
    for code in (original, proven):
        comments, strings = _literal_spans(code)
        masked = _blanked(code, comments + strings)
        if _brace_violations(masked) != []:
            return False
    return executable_part(original, lemma_names=lemma_names) == executable_part(proven, lemma_names=lemma_names)


__all__ = ["executable_part", "same_executable"]
