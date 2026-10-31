"""The method-header check for Vericoding answers (issue #74).

:func:`pcc.verifier.anti_cheat.anti_cheat` checks that the frozen task prefix still
occurs in the answer. An answer can keep those bytes and still declare a method with
a different header (a second declaration with an added clause, or the prefix echoed
inside a string literal), so :func:`header_violations` compares the header of every
declaration instead: each declaration of the task spec must appear exactly once in
the answer, with the same normalized header, directly followed by the ``{`` of its
body. Declarations only the answer has (helper lemmas, functions) are allowed.
"""

from __future__ import annotations

import re

from pcc.verifier.anti_cheat import (
    _BRACE_ATTR_OPEN_RE,
    _blanked,
    _literal_spans,
    _match_brace,
    _normalize,
)

#: Words that end a header search: a new declaration starts, so the declaration under
#: inspection is body-less (a body-less answer declaration is a header change).
_STOP_WORDS = frozenset(
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
    }
)

_DECL_RE = re.compile(
    r"(?<![A-Za-z0-9_'])"
    r"(?:(?:ghost|static|opaque|twostate)\s+)*"
    r"(?:(?:least|greatest)\s+)?"
    r"(?:function\s+method|method|function|predicate|lemma)"
    r"(?![A-Za-z0-9_'])"
)

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")


def header_violations(answer: str, task_spec: str) -> tuple[str, ...]:
    """Return one violation per declaration of ``task_spec`` whose header changed.

    The declarations of ``task_spec`` are every ``method``, ``function``,
    ``predicate`` and ``lemma``, with modifiers (``ghost``, ``static``, ``opaque``).
    The header runs from the first modifier or keyword to the ``{`` that opens the
    body, or to the end of the declaration when ``task_spec`` has no body for it (an
    optional trailing ``;`` excluded). A ``{`` that starts an attribute (``{:``) is
    part of the header, never the body. Headers compare with the scanner and
    normalizer of :mod:`pcc.verifier.anti_cheat`: comments removed, whitespace runs
    collapsed to one space.

    For each declaration of ``task_spec``, in ``task_spec`` order, the answer must
    declare the same name exactly once, with the same normalized header, followed
    directly by the ``{`` of its body (only whitespace and comments may sit between
    them; an added attribute counts as a change). Otherwise the violation is
    ``"header changed: <name>"``, or ``"header missing: <name>"`` when the answer
    does not declare the name at all. Two or more declarations with the same name
    count as one ``"header changed: <name>"``, even when one header matches, and so
    does a declaration whose body never opens (malformed or unbalanced braces).
    Declarations only the answer has are allowed.
    """
    spec_decls = _declarations(task_spec)
    if not spec_decls:
        return ()
    answer_decls = _declarations(answer)
    by_name: dict[str, list[tuple[str, bool]]] = {}
    for name, header, has_body in answer_decls:
        by_name.setdefault(name, []).append((header, has_body))
    violations: list[str] = []
    for name, header, _ in spec_decls:
        candidates = by_name.get(name, [])
        if not candidates:
            violations.append(f"header missing: {name}")
        elif len(candidates) != 1 or candidates[0][0] != header or not candidates[0][1]:
            violations.append(f"header changed: {name}")
    return tuple(violations)


def _read_word(text: str, pos: int) -> str:
    end = pos
    while end < len(text) and (text[end].isalnum() or text[end] in "_'"):
        end += 1
    return text[pos:end]


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _read_name(cleaned: str, pos: int) -> str | None:
    """The declared name after a declaration keyword, or ``None`` when absent."""
    match = _NAME_RE.match(cleaned, _skip_ws(cleaned, pos))
    return match.group(0) if match is not None else None


def _find_body_open(cleaned: str, start: int) -> int | None:
    """The ``{`` that opens the body of the declaration starting before ``start``.

    The first brace group at the top level of the header is the body, unless a new
    declaration (or the end of the enclosing scope) starts first (then the
    declaration is body-less) or the group is followed by more header material such
    as ``;`` (then it was an expression such as a set display). A ``{`` that starts
    an attribute (``{:``) is skipped as part of the header. An unclosed ``{`` is
    returned as the body open; the caller treats the missing close as no body.
    """
    depth_paren = 0
    depth_brack = 0
    pos = start
    size = len(cleaned)
    while pos < size:
        char = cleaned[pos]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        elif char == "}" and depth_paren == 0 and depth_brack == 0:
            return None
        elif char == "{" and depth_paren == 0 and depth_brack == 0:
            if _BRACE_ATTR_OPEN_RE.match(cleaned, pos) is not None:
                closing = _match_brace(cleaned, pos)
                if closing is None:
                    return None
                pos = closing + 1
                continue
            closing = _match_brace(cleaned, pos)
            if closing is None:
                return pos
            after = _skip_ws(cleaned, closing + 1)
            if after >= size or cleaned[after] == "}":
                return pos
            if cleaned[after] == "{":
                pos = after
                continue
            if _read_word(cleaned, after) in _STOP_WORDS:
                return pos
            pos = closing + 1
            continue
        elif depth_paren == 0 and depth_brack == 0 and (char.isalnum() or char in "_'"):
            word = _read_word(cleaned, pos)
            if word in _STOP_WORDS:
                return None
            pos += len(word)
            continue
        pos += 1
    return None


def _declarations(source: str) -> list[tuple[str, str, bool]]:
    """Parse ``source`` into ``(name, normalized header, has body)`` declarations."""
    comments, strings = _literal_spans(source)
    cleaned = _blanked(source, comments + strings)
    found: list[tuple[int, int, str]] = []
    for match in _DECL_RE.finditer(cleaned):
        name = _read_name(cleaned, match.end())
        if name is not None:
            found.append((match.start(), match.end(), name))
    decls: list[tuple[str, str, bool]] = []
    index = 0
    consumed = 0
    while index < len(found):
        start, end, name = found[index]
        if start < consumed:
            index += 1
            continue
        body_open = _find_body_open(cleaned, end)
        if body_open is not None:
            closing = _match_brace(cleaned, body_open)
            consumed = closing + 1 if closing is not None else len(cleaned)
            decls.append((name, _normalize(source[start:body_open]), closing is not None))
        else:
            following = found[index + 1][0] if index + 1 < len(found) else len(source)
            header = source[start:following].rstrip().removesuffix(";")
            decls.append((name, _normalize(header), False))
        index += 1
    return decls


__all__ = ["header_violations"]
