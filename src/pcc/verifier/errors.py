"""Parsing Dafny's error output into ``DafnyError`` tuples.

Used by ``resolve`` in ``pcc.verifier.dafny`` now and by ``verify`` (issue #20) later.
"""

from __future__ import annotations

import re
from dataclasses import replace

from pcc.core import DafnyError

#: A Dafny diagnostic line, e.g. ``post.dfy(3,0): Error: <msg>`` or
#: ``post.dfy(2,12): Related location: <msg>``. The file part matches any ``.dfy`` path, including
#: Windows paths such as ``C:\\proj\\main.dfy``.
_ERROR_LINE = re.compile(
    r"^(?P<path>.+\.dfy)\((?P<line>\d+),(?P<column>\d+)\): (?P<kind>Error|Related location): (?P<message>.*)$"
)

#: A source line in Dafny's ``|`` snippet, e.g. ``2 |   ensures r > x``. Separator (``  |``) and
#: marker (``  | ^``) lines carry no line number and do not match.
_CODE_LINE = re.compile(r"^\s*\d+\s*\|(?P<code>.*)$")

_PYLIB_NAME = "PyLib.dfy"


def parse_errors(output: str) -> tuple[DafnyError, ...]:
    """Parse Dafny's combined output into one ``DafnyError`` per ``Error`` line.

    A ``Related location`` line belongs to the previous error: it is appended to that error's
    ``related`` as ``<file>(<line>,<col>): <msg>``, followed by the source line it points to (the
    first ``<n> | <code>`` snippet line after it). A related location without a previous error is
    ignored. Every other line (source snippets of the error itself, ``|`` markers, blank lines,
    Dafny's summary lines) carries no error and is skipped, so Windows (``C:\\...``) paths and CRLF
    line endings need no special handling.

    Errors reported in ``PyLib.dfy`` get ``"PyLib.dfy: "`` in front of the message, so a caller sees
    at a glance that the Python library (not the checked code) failed to resolve.
    """
    errors: list[DafnyError] = []
    waiting_for_source = False
    for line in output.splitlines():
        match = _ERROR_LINE.match(line)
        if match is not None:
            waiting_for_source = False
            path = match.group("path")
            line_no = int(match.group("line"))
            column = int(match.group("column"))
            message = match.group("message")
            if match.group("kind") == "Related location":
                if not errors:
                    continue
                related = f"{path}({line_no},{column}): {message}"
                previous = errors[-1]
                combined = related if not previous.related else f"{previous.related}\n{related}"
                errors[-1] = replace(previous, related=combined)
                waiting_for_source = True
            else:
                if path.replace("\\", "/").rsplit("/", 1)[-1] == _PYLIB_NAME:
                    message = f"{_PYLIB_NAME}: {message}"
                errors.append(DafnyError(line_no, column, message))
            continue
        if waiting_for_source:
            code = _CODE_LINE.match(line)
            if code is not None:
                source = code.group("code")
                if source.startswith(" "):
                    source = source[1:]
                previous = errors[-1]
                errors[-1] = replace(previous, related=f"{previous.related}\n{source}")
                waiting_for_source = False
    return tuple(errors)


__all__ = ["parse_errors"]
