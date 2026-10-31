"""The ghost-only check on proof attempts."""

from __future__ import annotations

import re
import warnings
from collections.abc import Collection, Sequence

from pcc.verifier.anti_cheat import (
    _BRACE_ATTR_OPEN_RE,
    _blanked,
    _brace_violations,
    _literal_spans,
    _match_brace,
    _match_paren,
    _normalize,
)

_CLAUSE_STARTS = ("invariant", "decreases", "modifies")
_CLAUSE_KEYWORDS = frozenset({"invariant", "decreases", "modifies", "requires", "ensures", "reads"})
_LEMMA_RE = re.compile(r"(?<![A-Za-z0-9_'])(?:(?:ghost|static|opaque)\s+)*lemma(?![A-Za-z0-9_'])")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")

#: Words that end a lemma header search: a new declaration starts, so the lemma
#: under inspection is body-less (or its header ended). Modifiers (``ghost``,
#: ``static``, ``opaque``, ``twostate``, ``least``, ``greatest``) count too,
#: since they precede the next declaration keyword.
_DECL_STOP_WORDS = frozenset(
    {
        "method",
        "function",
        "predicate",
        "lemma",
        "copredicate",
        "class",
        "trait",
        "datatype",
        "codatatype",
        "newtype",
        "type",
        "module",
        "import",
        "const",
        "var",
        "constructor",
        "ghost",
        "static",
        "opaque",
        "twostate",
        "least",
        "greatest",
    }
)

_DECL_RE = re.compile(
    r"(?<![A-Za-z0-9_'])"
    r"(?:(?:ghost|static|opaque|twostate)\s+)*"
    r"(?:(?:least|greatest)\s+)?"
    r"(?:function\s+method|method|function|predicate|lemma)"
    r"(?![A-Za-z0-9_'])"
)

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


def _header_spans(source: str) -> list[tuple[int, int]]:
    """The ``(start, body open)`` spans of method/function/predicate/lemma headers.

    ``start`` is the first modifier (or keyword) and the span ends at the ``{``
    opening the body, or where the header scan stopped for a body-less
    declaration. Matches are found only outside strings and comments.
    """
    comments, strings = _literal_spans(source)
    masked = _blanked(source, comments + strings)
    spans: list[tuple[int, int]] = []
    for match in _DECL_RE.finditer(masked):
        name = _read_lemma_name(masked, match.end())
        if name is None:
            continue
        name_end = _skip_header_junk(masked, match.end()) + len(name)
        body_open, header_end = _find_lemma_body_open(masked, name_end)
        spans.append((match.start(), body_open if body_open is not None else header_end))
    return spans


def _strip_clause_spans(masked: str, headers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Clause spans for :func:`strip_ghost`: ``decreases`` on a method header is kept.

    Loop ``decreases`` clauses sit after the ``{`` opening the body, hence
    outside every header span, and are still stripped; so are ``invariant`` and
    ``modifies`` clauses everywhere.
    """
    spans: list[tuple[int, int]] = []
    for keyword in _CLAUSE_STARTS:
        for match in _word_pattern(keyword).finditer(masked):
            end = _clause_end(masked, match.end())
            if end <= match.start():
                continue
            if keyword == "decreases" and any(start <= match.start() and end <= stop for start, stop in headers):
                continue
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


def _skip_header_junk(masked: str, pos: int) -> int:
    """Skip whitespace and ``{:...}`` attribute groups starting at ``pos``."""
    while True:
        pos = _skip_ws(masked, pos)
        if _BRACE_ATTR_OPEN_RE.match(masked, pos) is not None:
            closing = _match_brace(masked, pos)
            if closing is None:
                return pos
            pos = closing + 1
            continue
        return pos


def _read_lemma_name(masked: str, pos: int) -> str | None:
    """The declared name after a ``lemma`` keyword, or ``None`` when absent."""
    match = _NAME_RE.match(masked, _skip_header_junk(masked, pos))
    return match.group(0) if match is not None else None


def _find_lemma_body_open(masked: str, start: int) -> tuple[int | None, int]:
    """The ``(body_open, header_end)`` of the lemma declaration starting before ``start``.

    ``body_open`` is the ``{`` that opens the lemma body, or ``None`` when the
    lemma is body-less (or malformed); ``header_end`` is where the header scan
    stopped. The first brace group at the top level of the header is the body,
    unless a new declaration starts first or a ``;`` ends the header (then the
    lemma is body-less), or the group is followed by more header material such
    as ``;`` (then it was an expression such as a set display). A ``{`` that
    starts an attribute (``{:``) is part of the header, never the body.
    """
    depth_paren = 0
    depth_brack = 0
    pos = start
    size = len(masked)
    while pos < size:
        char = masked[pos]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        elif char == "}" and depth_paren == 0 and depth_brack == 0:
            return None, pos
        elif char == "{" and depth_paren == 0 and depth_brack == 0:
            if _BRACE_ATTR_OPEN_RE.match(masked, pos) is not None:
                closing = _match_brace(masked, pos)
                if closing is None:
                    return None, pos
                pos = closing + 1
                continue
            closing = _match_brace(masked, pos)
            if closing is None:
                return pos, pos
            after = _skip_ws(masked, closing + 1)
            if after >= size or masked[after] == "}":
                return pos, pos
            if masked[after] == "{":
                pos = after
                continue
            if _read_word(masked, after) in _DECL_STOP_WORDS:
                return pos, pos
            pos = closing + 1
            continue
        elif char == ";" and depth_paren == 0 and depth_brack == 0:
            return None, pos + 1
        elif depth_paren == 0 and depth_brack == 0 and (char.isalnum() or char in "_'"):
            word = _read_word(masked, pos)
            if word in _DECL_STOP_WORDS:
                return None, pos
            pos += len(word)
            continue
        pos += 1
    return None, size


def _lemma_declarations(source: str) -> list[tuple[str, int, int]]:
    """Parse ``source`` into ``(name, start, end)`` lemma declarations.

    Declaration names are found only outside strings and comments, and a lemma
    body is skipped as one balanced span, so nested braces never create
    declarations. Modifiers (``ghost``, ``static``, ``opaque``) are part of the
    match; the span runs from the first modifier (or keyword) through the body,
    or through the header end for a body-less lemma. The function never raises.
    """
    comments, strings = _literal_spans(source)
    masked = _blanked(source, comments + strings)
    found: list[tuple[str, int, int]] = []
    consumed = 0
    for match in _LEMMA_RE.finditer(masked):
        if match.start() < consumed:
            continue
        name = _read_lemma_name(masked, match.end())
        if name is None:
            continue
        name_end = _skip_header_junk(masked, match.end()) + len(name)
        body_open, header_end = _find_lemma_body_open(masked, name_end)
        if body_open is not None:
            closing = _match_brace(masked, body_open)
            end = closing + 1 if closing is not None else len(source)
        else:
            end = header_end
        end = max(end, match.end())
        consumed = max(consumed, end)
        found.append((name, match.start(), end))
    return found


def _wanted_lemma_names(dafny: str, lemma_names: Collection[str]) -> list[str]:
    """``lemma_names`` plus every name declared in ``dafny``, deduplicated in first-seen order."""
    return list(dict.fromkeys([*lemma_names, *(name for name, _, _ in _lemma_declarations(dafny))]))


def _lemma_declaration_spans(source: str, lemma_names: Collection[str]) -> list[tuple[int, int]]:
    """The spans of the lemma declarations whose names are in ``lemma_names``."""
    wanted = set(lemma_names)
    return [(start, end) for name, start, end in _lemma_declarations(source) if name in wanted]


def declared_lemma_names(dafny: str) -> tuple[str, ...]:
    """The lemma names declared in ``dafny``, deduplicated in first-seen order.

    Only ``lemma`` keywords outside strings and comments count, with optional
    ``ghost``, ``static`` or ``opaque`` modifiers; a lemma body is skipped as
    one balanced span, so nested braces never create declarations.
    """
    return tuple(dict.fromkeys(name for name, _, _ in _lemma_declarations(dafny)))


def spec_referenced_lemmas(dafny: str, lemma_names: Collection[str]) -> tuple[str, ...]:
    """The subset of ``lemma_names`` referenced by kept (non-ghost) code.

    Kept code is ``dafny`` with every ghost span and every lemma declaration
    removed; a name occurring there as a word is referenced by a kept method
    header or specification. Names declared in ``dafny`` but missing from
    ``lemma_names`` are checked too.
    """
    wanted = _wanted_lemma_names(dafny, lemma_names)
    if not wanted:
        return ()
    comments, strings = _literal_spans(dafny)
    masked = _blanked(dafny, comments + strings)
    spans: list[tuple[int, int]] = []
    spans.extend(_strip_clause_spans(masked, _header_spans(dafny)))
    spans.extend(_assert_spans(masked))
    spans.extend(_calc_spans(masked))
    spans.extend(_reveal_spans(masked))
    spans.extend(_ghost_var_spans(masked))
    spans.extend(_lemma_spans(masked, wanted))
    spans.extend(_lemma_declaration_spans(dafny, wanted))
    kept = _blanked(masked, _merge_spans(spans))
    return tuple(name for name in wanted if name != "" and _word_pattern(name).search(kept) is not None)


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
    declarations, lemma declarations whose names are in ``lemma_names`` and bare
    calls ``Name(args);`` whose ``Name`` is in ``lemma_names``. Everything else
    (``requires``, ``ensures``, ``reads``, ``ghost const`` and friends) is kept,
    so unhandled additions compare unequal. String and character literals are
    kept verbatim; only comments are removed. The function never raises.
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
    spans.extend(_lemma_declaration_spans(dafny_method, lemma_names))
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


def strip_ghost(dafny: str, *, lemma_names: Collection[str]) -> str:
    """Remove ghost code and lemma declarations, keeping character positions.

    This removes the same ghost code as :func:`executable_part` (ghost clauses,
    asserts, ``calc`` blocks, ``reveal`` statements, ``ghost var`` declarations
    and bare calls to ``lemma_names``) and also every lemma declaration, but it
    keeps lines, indentation and comments of the rest: every removed character
    except newlines becomes a space, so the output has the same length and the
    same newlines as the input. Method headers stay unchanged, including their
    ``requires``, ``ensures`` and ``decreases`` clauses. Names declared in
    ``dafny`` but missing from ``lemma_names`` are stripped too.

    A lemma referenced by kept (non-ghost) code, such as a method header or
    specification, is retained and produces a ``UserWarning`` naming it; every
    other lemma declaration and every bare call to a lemma is stripped. The
    function never raises, and for every input
    ``executable_part(strip_ghost(x)) == executable_part(x)``.
    """
    wanted = _wanted_lemma_names(dafny, lemma_names)
    retained = set(spec_referenced_lemmas(dafny, wanted))
    comments, strings = _literal_spans(dafny)
    masked = _blanked(dafny, comments + strings)
    headers = _header_spans(dafny)
    spans: list[tuple[int, int]] = []
    spans.extend(_strip_clause_spans(masked, headers))
    spans.extend(_assert_spans(masked))
    spans.extend(_calc_spans(masked))
    spans.extend(_reveal_spans(masked))
    spans.extend(_ghost_var_spans(masked))
    spans.extend(_lemma_spans(masked, wanted))
    for name, start, end in _lemma_declarations(dafny):
        if name in wanted and name not in retained:
            spans.append((start, end))
    chars = list(dafny)
    for start, end in _merge_spans(spans):
        for index in range(start, min(end, len(chars))):
            if chars[index] != "\n":
                chars[index] = " "
    for name in wanted:
        if name in retained:
            warnings.warn(
                f"lemma {name} is referenced in a specification; keeping its declaration",
                UserWarning,
                stacklevel=2,
            )
    return "".join(chars)


__all__ = [
    "declared_lemma_names",
    "executable_part",
    "same_executable",
    "spec_referenced_lemmas",
    "strip_ghost",
]
