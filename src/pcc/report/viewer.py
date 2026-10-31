"""Local results viewer: serve a scan file as a browsable page (issue #67).

The viewer only reads result files and starts no runs: it calls no model, no
sandbox and no pipeline step. Only the standard library and ``pcc.core`` are
used, so the page can be served while every outside system is unreachable.
"""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pcc.core import Result, ToolError, from_data, to_data


@dataclass(frozen=True, slots=True)
class Entry:
    """One browsable row: the project name and its pipeline result."""

    project: str
    result: Result


def load_entries(path: Path) -> tuple[Entry, ...]:
    """Read every non-empty line of ``path`` before any server starts.

    Blank lines are skipped; every other line must be a JSON object with a
    string ``project`` and a ``result`` that deserializes into ``Result``.
    Extra keys on the line are ignored, so evaluation runs that record more
    per line still load. Physical lines are counted from 1, including blank
    ones, so an error names the line as seen in the file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot read {path}: {exc}") from exc
    entries: list[Entry] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except ValueError as exc:
            raise ToolError(f"{path}: line {lineno}: not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or "project" not in data or "result" not in data:
            raise ToolError(f"{path}: line {lineno}: expected an object with project and result")
        project = data["project"]
        if not isinstance(project, str):
            raise ToolError(f"{path}: line {lineno}: project must be a string")
        try:
            result = from_data(Result, data["result"])
        except (TypeError, ValueError) as exc:
            raise ToolError(f"{path}: line {lineno}: not a valid result: {exc}") from exc
        entries.append(Entry(project, result))
    return tuple(entries)


def page_html() -> str:
    """The viewer page, loaded from the ``viewer.html`` file next to this module."""
    return importlib.resources.files("pcc.report").joinpath("viewer.html").read_text(encoding="utf-8")


def results_payload(entries: tuple[Entry, ...]) -> list[dict[str, Any]]:
    """The ``GET /api/results`` body: one ``project``/``result`` object per entry."""
    return [{"project": entry.project, "result": to_data(entry.result)} for entry in entries]


def create_server(entries: tuple[Entry, ...], port: int) -> ThreadingHTTPServer:
    """A ``ThreadingHTTPServer`` bound to ``127.0.0.1`` only, serving ``entries``.

    ``GET /`` returns the page, ``GET /api/results`` returns the JSON list;
    any other path is 404 and any other method is 405. ``port`` 0 picks a free
    port; the bound address is available as ``server.server_address``.
    """
    page = page_html().encode("utf-8")
    payload = json.dumps(results_payload(entries), ensure_ascii=False, sort_keys=True).encode("utf-8")

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send_bytes(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/results":
                self._send_bytes(200, payload, "application/json")
            else:
                self.send_error(404, "Not found")

        def _deny(self) -> None:
            self.send_error(405, "Method not allowed")

        do_POST = _deny
        do_PUT = _deny
        do_DELETE = _deny
        do_PATCH = _deny
        do_HEAD = _deny
        do_OPTIONS = _deny
        do_TRACE = _deny
        do_CONNECT = _deny

        def __getattr__(self, name: str) -> Any:
            # Any method beyond the ones above (custom verbs included) is not
            # allowed either; without this fallback the base class answers 501.
            if name.startswith("do_"):
                return self._deny
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

        def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            # Keep the command output to the single URL line; request logs would
            # only clutter the terminal of a local viewer.
            pass

        def version_string(self) -> str:
            return "PCCViewer"

    return ThreadingHTTPServer(("127.0.0.1", port), ViewerHandler)


def serve(entries: tuple[Entry, ...], port: int) -> None:
    """Serve ``entries`` on ``port`` until Ctrl-C; the caller maps that to exit 0."""
    with create_server(entries, port) as server:
        server.serve_forever()


__all__ = ["Entry", "create_server", "load_entries", "page_html", "results_payload", "serve"]
