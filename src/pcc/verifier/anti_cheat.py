"""The anti-cheat scan over Dafny proof attempts.

Dafny 4.11 verifies some cheats with exit code 0 (``{:axiom}``, ``@Axiom``, ``expect``,
``decreases *`` and ``{:selective_checking}``), so the exit code alone cannot catch them.
:func:`anti_cheat` scans the source instead.
"""

from __future__ import annotations

import bisect
import re

ALLOWED_ATTRIBUTES = {
    "trigger",
    "induction",
    "fuel",
    "Fuel",
    "opaque",
    "split_here",
    "isolate_assertions",
    "IsolateAssertions",
}

_MEMBER_FORMS = {
    "lemma": "body-less lemma",
    "function": "body-less function",
    "method": "body-less method",
}

_RESOURCE_BRACE_NAMES = {"rlimit", "resource_limit", "timelimit", "timelimitmultiplier"}
_RESOURCE_AT_NAMES = {"resourcelimit", "timelimit", "timelimitmultiplier"}

#: Words that start a new declaration. Finding one while looking for a member body (or right
#: after a candidate body group) means the member ended; the search stops there.
_DECLARATION_WORDS = frozenset(
    {
        "lemma",
        "function",
        "method",
        "predicate",
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

_MEMBER_RE = re.compile(r"(?<![A-Za-z0-9_'])(lemma|function|method)(?![A-Za-z0-9_'])")
_KEYWORD_RE = {
    "assume": re.compile(r"(?<![A-Za-z0-9_'])assume(?![A-Za-z0-9_'])"),
    "expect": re.compile(r"(?<![A-Za-z0-9_'])expect(?![A-Za-z0-9_'])"),
    "decreases *": re.compile(r"(?<![A-Za-z0-9_'])decreases(?![A-Za-z0-9_'])\s*\*(?![A-Za-z0-9_'])"),
}
_AT_ATTR_RE = re.compile(r"(?<![A-Za-z0-9_'])@([A-Za-z_][A-Za-z0-9_]*)")
_BRACE_ATTR_OPEN_RE = re.compile(r"\{\s*:")
_ATTR_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']*")
_ATTR_SECOND_RE = re.compile(r"\s*([A-Za-z][A-Za-z0-9_']*)")
_MODULE_KW_RE = re.compile(r"(?<![A-Za-z0-9_'])module(?![A-Za-z0-9_'])")
_MODULE_NAME_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_']*)")
_MODULE_REFINES_RE = re.compile(r"\s*refines\s+[A-Za-z_][A-Za-z0-9_.']*")
_MODULE_OPEN_RE = re.compile(r"\s*\{")

#: An attribute occurrence: ``(start, end, style, name, second_word)``. ``style`` is
#: ``"brace"`` for ``{:name ...}`` and ``"at"`` for ``@Name(...)``; ``second_word`` is the
#: word after the name inside braces (as in ``{:verify false}``), else ``""``.
_Attribute = tuple[int, int, str, str, str]


def anti_cheat(dafny_code: str, frozen_spec: str, *, assumptions_module: str = "") -> tuple[str, ...]:
    """Return every violation in ``dafny_code``; an empty tuple means it passes.

    Outside ``assumptions_module`` every one of these forms is a violation (an empty
    ``assumptions_module`` means no module may hold contracts):

    - ``assume`` statements, with or without attributes (``assume {:axiom} false;`` included);
    - ``expect`` statements, which Dafny 4.11 only checks at run time;
    - ``{:axiom}`` and ``@Axiom`` in any position;
    - body-less lemmas, functions and methods (Dafny warns on them, and they are contracts
      in disguise);
    - ``{:verify false}``, which skips the proof (Dafny 4.11 has no ``@`` spelling of it);
    - ``{:only}`` and ``@VerifyOnly``, which disable the verification of every other member;
    - ``{:extern}``, a body-less method that Dafny takes on trust;
    - ``decreases *``, which skips the termination check;
    - ``{:termination false}``, which claims the function needs no termination proof;
    - any attribute that overrides the resource limit: ``{:rlimit}``, ``{:resource_limit}``,
      ``{:timeLimit}``, ``{:timeLimitMultiplier}``, ``@ResourceLimit``, ``@TimeLimit`` and
      ``@TimeLimitMultiplier``;
    - any other attribute outside a fixed allowlist (``trigger``, ``induction``, ``fuel`` /
      ``Fuel``, ``opaque``, ``split_here``, ``isolate_assertions`` / ``IsolateAssertions``),
      with or without arguments, reported as ``attribute <name>``: some, like
      ``{:selective_checking}``, make Dafny accept ``ensures false`` with exit code 0;
    - any change to ``frozen_spec``: the frozen spec must still appear unchanged in
      ``dafny_code``;
    - unbalanced braces.

    Each violation is ``"line <n>: <form>"``; violations sort by line, and several on one
    line by column. Attributes are checked against the allowlist everywhere, the assumptions
    module included (where ``{:axiom}`` / ``@Axiom`` contracts are additionally allowed):
    every ``{:name ...}`` and every ``@Name``, with or without arguments, whose name is not
    in ``ALLOWED_ATTRIBUTES`` is a violation ``"attribute <name>"``. An ``@`` directly after
    an identifier character (``old@L(e)``, ``unchanged@L``) is a label, not an attribute.
    ``{:axiom}`` / ``@Axiom`` keep the form ``axiom``, and the resource limit attributes keep
    ``resource limit override``. An attribute that already has its own form (``{:verify
    false}``, ``{:termination false}``, ``{:only}``, ``@VerifyOnly``, ``{:extern}``) is
    reported with that form only, not also as ``attribute <name>``.

    Inside the assumptions module, a body-less lemma, function or method marked ``{:axiom}``
    or ``@Axiom`` is allowed. Outside it, ``{:axiom}`` or ``@Axiom`` is reported as ``axiom``,
    and a body-less lemma, function or method as ``body-less lemma``, ``body-less function``
    or ``body-less method``.

    An empty ``frozen_spec`` skips the spec check. Scanning happens after comments (``//``,
    ``/* */`` nested) and string and char literals are removed, so ``"assume"`` inside a
    string or a comment is no violation. A ``'`` starts a char literal only where it does not
    follow an identifier character, since Dafny identifiers may contain ``'`` (``s'``).
    Module bounds are found by matching braces after ``module <Name>``. The frozen spec check
    collapses whitespace runs to one space and removes comments on both sides.
    """
    comments, strings = _literal_spans(dafny_code)
    cleaned = _blanked(dafny_code, comments + strings)
    line_starts = _line_starts(dafny_code)
    violations: list[tuple[int, int, str]] = []

    def position(index: int) -> tuple[int, int]:
        line = bisect.bisect_right(line_starts, index)
        return line, index - line_starts[line - 1] + 1

    attributes = _find_attributes(cleaned)
    module_spans = _module_spans(cleaned, assumptions_module) if assumptions_module else []

    def inside_assumptions(index: int) -> bool:
        return any(start <= index < end for start, end in module_spans)

    suppressed_axioms: set[int] = set()
    for match in _MEMBER_RE.finditer(cleaned):
        kind = match.group(1)
        if kind == "method" and _preceded_by_compound(match.start(), cleaned):
            continue
        attached = _attached_axiom_indices(cleaned, attributes, match.start(), match.end())
        is_inside = inside_assumptions(match.start())
        if is_inside:
            suppressed_axioms.update(attached)
        if not _has_body(cleaned, match.end()) and not (is_inside and attached):
            form = _MEMBER_FORMS[kind]
            violations.append((*position(match.start()), form))

    for index, (start, _end, style, name, second) in enumerate(attributes):
        attr_form = _attribute_form(style, name, second)
        if attr_form is None:
            continue
        if attr_form == "axiom" and index in suppressed_axioms:
            continue
        violations.append((*position(start), attr_form))

    for form, pattern in _KEYWORD_RE.items():
        for match in pattern.finditer(cleaned):
            violations.append((*position(match.start()), form))

    for bad in _brace_violations(cleaned):
        violations.append((*position(bad), "unbalanced braces"))

    if frozen_spec != "" and _normalize(frozen_spec) not in _normalize(dafny_code):
        violations.append((0, 0, "frozen spec changed"))

    violations.sort()
    return tuple(f"line {line}: {form}" for line, _column, form in violations)


def _is_ident_char(char: str) -> bool:
    return char.isalnum() or char in "_'"


def _literal_spans(code: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return ``(comment_spans, string_spans)``; spans are half-open and never overlap."""
    comments: list[tuple[int, int]] = []
    strings: list[tuple[int, int]] = []
    i = 0
    n = len(code)
    while i < n:
        if code.startswith("//", i):
            end = code.find("\n", i)
            if end == -1:
                end = n
            comments.append((i, end))
            i = end
        elif code.startswith("/*", i):
            depth = 0
            j = i
            while j < n:
                if code.startswith("/*", j):
                    depth += 1
                    j += 2
                elif code.startswith("*/", j):
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            comments.append((i, j))
            i = j
        elif code[i] == '"' and not (i > 0 and code[i - 1] == "@"):
            j = i + 1
            while j < n and code[j] != "\n":
                if code[j] == "\\":
                    j += 2
                elif code[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            strings.append((i, j))
            i = j
        elif code[i] == '"' and i > 0 and code[i - 1] == "@":
            j = i + 1
            while j < n:
                if code.startswith('""', j):
                    j += 2
                elif code[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            strings.append((i, j))
            i = j
        elif code[i] == "'" and (i == 0 or not _is_ident_char(code[i - 1])):
            j = i + 1
            while j < n and code[j] != "\n":
                if code[j] == "\\":
                    j += 2
                elif code[j] == "'":
                    j += 1
                    break
                else:
                    j += 1
            strings.append((i, j))
            i = j
        else:
            i += 1
    return comments, strings


def _blanked(code: str, spans: list[tuple[int, int]]) -> str:
    """Replace every span with spaces, keeping newlines so positions still line up."""
    chars = list(code)
    for start, end in spans:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _line_starts(code: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", code):
        starts.append(match.end())
    return starts


def _match_brace(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _match_paren(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _match_paren_back(text: str, closing: int) -> int | None:
    depth = 0
    for index in range(closing, -1, -1):
        if text[index] == ")":
            depth += 1
        elif text[index] == "(":
            depth -= 1
            if depth == 0:
                return index
    return None


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _skip_ws_back(text: str, pos: int) -> int:
    while pos > 0 and text[pos - 1].isspace():
        pos -= 1
    return pos


def _skip_ws_and_attributes(text: str, pos: int) -> int:
    """Skip whitespace and attribute groups (``{:...}`` and ``@Name(...)``)."""
    while True:
        pos = _skip_ws(text, pos)
        if _BRACE_ATTR_OPEN_RE.match(text, pos) is not None:
            closing = _match_brace(text, pos)
            if closing is None:
                return pos
            pos = closing + 1
            continue
        at_match = _AT_ATTR_RE.match(text, pos)
        if at_match is not None:
            end = at_match.end()
            after = _skip_ws(text, end)
            if after < len(text) and text[after] == "(":
                closing = _match_paren(text, after)
                if closing is not None:
                    end = closing + 1
            pos = end
            continue
        return pos


def _read_word(text: str, pos: int) -> str:
    end = pos
    while end < len(text) and (text[end].isalnum() or text[end] in "_'"):
        end += 1
    return text[pos:end]


def _attribute_form(style: str, name: str, second: str) -> str | None:
    """Map an attribute to its violation form, or ``None`` when it is allowlisted."""
    lowered = name.lower()
    if lowered == "axiom":
        return "axiom"
    if style == "brace":
        if lowered == "verify" and second.lower() == "false":
            return "verify false"
        if lowered == "termination" and second.lower() == "false":
            return "termination false"
        if lowered == "only":
            return "only"
        if lowered == "extern":
            return "extern"
        if lowered in _RESOURCE_BRACE_NAMES:
            return "resource limit override"
    elif style == "at":
        if lowered == "verifyonly":
            return "only"
        if lowered in _RESOURCE_AT_NAMES:
            return "resource limit override"
    if name in ALLOWED_ATTRIBUTES:
        return None
    return f"attribute {name}"


def _find_attributes(cleaned: str) -> list[_Attribute]:
    """Find every ``{:name ...}`` and ``@Name`` occurrence with its span and name."""
    found: list[_Attribute] = []
    for match in _AT_ATTR_RE.finditer(cleaned):
        name = match.group(1)
        end = match.end()
        after = _skip_ws(cleaned, end)
        if after < len(cleaned) and cleaned[after] == "(":
            closing = _match_paren(cleaned, after)
            if closing is not None:
                end = closing + 1
        found.append((match.start(), end, "at", name, ""))
    pos = 0
    while True:
        opening = _BRACE_ATTR_OPEN_RE.search(cleaned, pos)
        if opening is None:
            break
        name_match = _ATTR_NAME_RE.match(cleaned, opening.end())
        if name_match is None:
            pos = opening.end()
            continue
        name = name_match.group(0)
        second_match = _ATTR_SECOND_RE.match(cleaned, name_match.end())
        second = second_match.group(1) if second_match is not None else ""
        closing = _match_brace(cleaned, opening.start())
        end = closing + 1 if closing is not None else name_match.end()
        found.append((opening.start(), end, "brace", name, second))
        pos = opening.end()
    found.sort()
    return found


def _brace_violations(cleaned: str) -> list[int]:
    """Return the positions of every unmatched closing brace and unclosed opener."""
    stack: list[int] = []
    bad: list[int] = []
    for index, char in enumerate(cleaned):
        if char == "{":
            stack.append(index)
        elif char == "}":
            if stack:
                stack.pop()
            else:
                bad.append(index)
    return bad + stack


def _module_spans(cleaned: str, name: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of every ``module <name> { ... }`` block."""
    spans: list[tuple[int, int]] = []
    for match in _MODULE_KW_RE.finditer(cleaned):
        name_match = _MODULE_NAME_RE.match(cleaned, match.end())
        if name_match is None or name_match.group(1) != name:
            continue
        rest = name_match.end()
        refines = _MODULE_REFINES_RE.match(cleaned, rest)
        if refines is not None:
            rest = refines.end()
        opening = _MODULE_OPEN_RE.match(cleaned, rest)
        if opening is None:
            continue
        closing = _match_brace(cleaned, opening.end() - 1)
        if closing is None:
            continue
        spans.append((match.start(), closing))
    return spans


def _preceded_by_compound(start: int, cleaned: str) -> bool:
    """Check whether a ``method`` keyword is the second half of ``function method``."""
    back = _skip_ws_back(cleaned, start)
    word_start = back
    while word_start > 0 and _is_ident_char(cleaned[word_start - 1]):
        word_start -= 1
    return cleaned[word_start:back] in ("function", "predicate")


def _has_body(cleaned: str, start: int) -> bool:
    """Check whether the member starting before ``start`` has a ``{ ... }`` body.

    The first brace group at the top level of the header is the body, unless a new
    declaration starts first (then the member is body-less) or the group is followed by
    more header material (then it was an expression such as a set display).
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
        elif char == "{" and depth_paren == 0 and depth_brack == 0:
            if _BRACE_ATTR_OPEN_RE.match(cleaned, pos) is not None:
                closing = _match_brace(cleaned, pos)
                pos = size if closing is None else closing + 1
                continue
            closing = _match_brace(cleaned, pos)
            if closing is None:
                return True
            after = _skip_ws_and_attributes(cleaned, closing + 1)
            if after >= size or cleaned[after] in "};":
                return True
            if cleaned[after] == "{":
                pos = after
                continue
            if _read_word(cleaned, after) in _DECLARATION_WORDS:
                return True
            pos = closing + 1
            continue
        elif depth_paren == 0 and depth_brack == 0 and (char.isalnum() or char in "_'"):
            word = _read_word(cleaned, pos)
            if word in _DECLARATION_WORDS:
                return False
            pos += len(word)
            continue
        pos += 1
    return False


def _attached_axiom_indices(cleaned: str, attributes: list[_Attribute], kind_start: int, kind_end: int) -> list[int]:
    """Return the indices of axiom attributes attached to the member at ``kind_start``.

    An attribute attaches forward when only whitespace and other attributes sit between
    the member keyword and it (``lemma {:axiom} L()``), and backward when only whitespace
    sits between it and the member keyword (``@Axiom`` on the line before the lemma).
    """
    by_start = {start: (index, end) for index, (start, end, _s, _n, _sec) in enumerate(attributes)}
    by_end = {end: (index, start) for index, (start, end, _s, _n, _sec) in enumerate(attributes)}
    attached: list[int] = []
    pos = _skip_ws(cleaned, kind_end)
    while pos in by_start:
        index, end = by_start[pos]
        if attributes[index][3].lower() == "axiom" and index not in attached:
            attached.append(index)
        pos = _skip_ws(cleaned, end)
    pos = kind_start
    while True:
        back = _skip_ws_back(cleaned, pos)
        if back not in by_end:
            break
        index, start = by_end[back]
        if attributes[index][3].lower() == "axiom" and index not in attached:
            attached.append(index)
        pos = start
    return attached


def _normalize(text: str) -> str:
    """Remove comments and collapse every whitespace run to one space."""
    comments, _strings = _literal_spans(text)
    return re.sub(r"\s+", " ", _blanked(text, comments)).strip()


__all__ = ["ALLOWED_ATTRIBUTES", "anti_cheat"]
