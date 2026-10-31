"""Tests for the local results viewer (``pcc-check view``, issue #67).

The server runs on port 0 in a thread and ``webbrowser.open`` is monkeypatched,
so the tests need no network and no browser.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import socket
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from pcc import models as models_module
from pcc import pipeline as pipeline_module
from pcc.cli import main
from pcc.core import Result, State, serialize
from pcc.report import viewer
from pcc.sandbox import run as sandbox_run_module

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scan.jsonl"

EXPECTED_STATES = (State.PROVEN, State.COUNTEREXAMPLE, State.OPEN, State.NOT_CHECKABLE)

ESCAPING_REASON = "<script>alert(1)</script>"


def _base(server: ThreadingHTTPServer) -> str:
    """The base URL of ``server`` from its bound address."""
    return f"http://127.0.0.1:{server.server_address[1]}"


def _get(url: str, method: str = "GET") -> tuple[int, str, bytes]:
    """GET ``url`` with ``method``; HTTP errors come back as ``(code, type, body)``."""
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.headers.get_content_type(), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read()


def _wait_ready(base: str) -> None:
    """Wait until the server at ``base`` answers ``GET /api/results`` with 200."""
    deadline = time.monotonic() + 10.0
    while True:
        try:
            status, _, _ = _get(base + "/api/results")
            if status == 200:
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        assert time.monotonic() < deadline, f"server at {base} did not become ready"
        time.sleep(0.05)


@pytest.fixture()
def live_server() -> Iterator[ThreadingHTTPServer]:
    """The fixture file served on port 0 in a thread, shut down after the test."""
    server = viewer.create_server(viewer.load_entries(FIXTURE), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=10)
    server.server_close()


def test_api_returns_four_results_with_four_states(live_server: ThreadingHTTPServer) -> None:
    status, content_type, body = _get(_base(live_server) + "/api/results")
    assert status == 200
    assert content_type == "application/json"
    payload = json.loads(body.decode("utf-8"))
    assert [entry["project"] for entry in payload] == ["demo-shop"] * 4
    results = [serialize.from_data(Result, entry["result"]) for entry in payload]
    assert [result.state for result in results] == list(EXPECTED_STATES)


def test_api_returns_escaping_reason_byte_identical(live_server: ThreadingHTTPServer) -> None:
    _, _, body = _get(_base(live_server) + "/api/results")
    assert ESCAPING_REASON in body.decode("utf-8")
    payload = json.loads(body.decode("utf-8"))
    assert payload[2]["result"]["reason"] == ESCAPING_REASON
    assert payload[2]["result"]["state"] == "open"


def test_api_returns_null_for_missing_proof_and_counterexample(live_server: ThreadingHTTPServer) -> None:
    _, _, body = _get(_base(live_server) + "/api/results")
    payload = json.loads(body.decode("utf-8"))
    assert payload[3]["result"]["state"] == "not_checkable"
    assert payload[3]["result"]["proof"] is None
    assert payload[3]["result"]["counterexample"] is None


def test_page_is_self_contained(live_server: ThreadingHTTPServer) -> None:
    status, content_type, body = _get(_base(live_server) + "/")
    assert status == 200
    assert content_type == "text/html"
    text = body.decode("utf-8")
    assert "<link" not in text
    assert "src=" not in text
    assert "innerHTML" not in text
    assert "textContent" in text
    for header in ("project", "path", "qualname", "state", "reason", "cost"):
        assert f"<th>{header}</th>" in text
    assert "–" in text  # noqa: RUF001 - the ticket mandates this placeholder for a null field
    for key in (
        "claims",
        "precondition",
        "predicate",
        "dafny_function",
        "dafny_spec",
        "counterexample",
        "second_opinion",
        "usage",
    ):
        assert key in text


PAGE_SCRIPT_MARKER = 'document.getElementById("state-filter").addEventListener'

_JS_DRIVER = """
"use strict";
const fs = require("fs");
const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
entries = payload;

const filterValues = { "state-filter": "", "project-filter": "", "search": "" };

function makeNode(tag) {
  return {
    tag: tag,
    children: [],
    textContent: "",
    value: "",
    className: "",
    hidden: false,
    firstChild: null,
    appendChild(child) {
      this.children.push(child);
      if (this.firstChild === null) { this.firstChild = child; }
      return child;
    },
    removeChild(child) {
      const at = this.children.indexOf(child);
      if (at >= 0) { this.children.splice(at, 1); }
      this.firstChild = this.children.length > 0 ? this.children[0] : null;
      return child;
    },
    addEventListener() {},
  };
}

const byId = {};
globalThis.document = {
  createElement: (tag) => makeNode(tag),
  getElementById: (id) => {
    if (!(id in byId)) {
      byId[id] = makeNode("#" + id);
      if (id in filterValues) { byId[id].value = filterValues[id]; }
    }
    return byId[id];
  },
};

function setFilters(state, project, query) {
  filterValues["state-filter"] = state;
  filterValues["project-filter"] = project;
  filterValues["search"] = query;
  for (const id of Object.keys(filterValues)) {
    if (id in byId) { byId[id].value = filterValues[id]; }
  }
}

function check(cond, label) {
  if (!cond) {
    console.error("JS_FAIL " + label);
    process.exit(1);
  }
  console.log("JS_OK " + label);
}

function texts(node, out) {
  out.push(node.textContent);
  for (const child of node.children) { texts(child, out); }
  return out;
}

check(show(null) === MISSING, "null-shows-missing");
check(show(undefined) === MISSING, "undefined-shows-missing");
check(show(0) === "0", "zero-renders");
check(show(false) === "false", "false-renders");
check(show("<script>alert(1)</script>") === "<script>alert(1)</script>", "escaping-unchanged");

check(costOf(entries[0].result).toFixed(2) === "0.50", "cost-proven");
check(costOf(entries[1].result).toFixed(2) === "0.20", "cost-counterexample");
check(costOf(entries[2].result).toFixed(2) === "0.00", "cost-open");
check(costOf(entries[3].result).toFixed(2) === "0.00", "cost-not-checkable");

setFilters("", "", "");
check(entries.every(matches), "all-match-without-filters");
setFilters("proven", "", "");
check(matches(entries[0]) && !matches(entries[1]), "state-filter-proven");
check(!matches(entries[2]) && !matches(entries[3]), "state-filter-rest");
setFilters("", "unknown-project", "");
check(entries.every((entry) => !matches(entry)), "project-filter-unknown");
setFilters("", "demo-shop", "");
check(entries.every(matches), "project-filter-known");
setFilters("", "", "ALERT");
check(matches(entries[2]) && !matches(entries[0]), "search-case-insensitive");
check(!matches(entries[1]) && !matches(entries[3]), "search-case-rest");
setFilters("", "", "legacy");
check(matches(entries[3]) && !matches(entries[0]), "search-path-qualname");

fillProjects();
check(byId["project-filter"].children.length === 1, "project-options");

setFilters("", "", "");
renderTable();
const rows = byId["rows"].children;
check(rows.length === 4, "table-has-four-rows");
const first = rows[0].children.map((cell) => cell.textContent).join("|");
check(first === "demo-shop|shop/cart.py|total|proven|verified|0.50", "table-first-row");
check(byId["summary"].textContent === "4 of 4 functions", "table-summary");

renderDetail(entries[3]);
const missing = texts(byId["detail"], []);
check(missing.includes(MISSING), "detail-null-shows-missing");
check(missing.includes("not_checkable"), "detail-state");
check(missing.includes("no type hints"), "detail-reason");
renderDetail(entries[2]);
const escaped = texts(byId["detail"], []);
check(escaped.includes("<script>alert(1)</script>"), "detail-escaping-as-text");
renderDetail(entries[0]);
const proven = texts(byId["detail"], []).join("\\n");
check(proven.includes("Returns the total price of the cart."), "detail-intent");
check(proven.includes("dafny_function"), "detail-dafny");
"""


def test_page_js_logic_runs_in_node(tmp_path: Path, live_server: ThreadingHTTPServer) -> None:
    """The page script executes: null placeholder, cost sums, filters and detail rendering.

    Node ships on the CI image; where it is missing the test skips and the static source
    assertions in ``test_page_is_self_contained`` still hold.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    status, _, body = _get(_base(live_server) + "/api/results")
    assert status == 200
    payload_path = tmp_path / "api.json"
    payload_path.write_text(body.decode("utf-8"), encoding="utf-8")
    html = viewer.page_html()
    assert PAGE_SCRIPT_MARKER in html
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    script = script.split(PAGE_SCRIPT_MARKER, 1)[0]
    driver = tmp_path / "viewer-driver.js"
    driver.write_text(script + _JS_DRIVER, encoding="utf-8")
    proc = subprocess.run([node, str(driver), str(payload_path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"node driver failed:\n{proc.stdout}\n{proc.stderr}"
    assert "JS_OK table-first-row" in proc.stdout
    assert "JS_OK detail-escaping-as-text" in proc.stdout


def test_unknown_path_returns_404(live_server: ThreadingHTTPServer) -> None:
    status, _, _ = _get(_base(live_server) + "/missing")
    assert status == 404


@pytest.mark.parametrize(
    ("method", "target"),
    (
        ("POST", "/"),
        ("POST", "/api/results"),
        ("POST", "/missing"),
        ("PUT", "/"),
        ("DELETE", "/api/results"),
        ("TRACE", "/"),
    ),
)
def test_non_get_methods_return_405(live_server: ThreadingHTTPServer, method: str, target: str) -> None:
    status, _, _ = _get(_base(live_server) + target, method=method)
    assert status == 405


def test_port_zero_binds_loopback_only() -> None:
    server = viewer.create_server((), 0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_empty_file_serves_empty_list(tmp_path: Path) -> None:
    empty = tmp_path / "scan.jsonl"
    empty.write_text("", encoding="utf-8")
    assert viewer.load_entries(empty) == ()
    server = viewer.create_server((), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _wait_ready(_base(server))
        status, _, body = _get(_base(server) + "/api/results")
        assert status == 200
        assert json.loads(body.decode("utf-8")) == []
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def test_load_entries_ignores_extra_line_keys(tmp_path: Path) -> None:
    """Lines in the shape of the evaluation runs carry extra keys next to ``project``/``result``."""
    line = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    line["task_id"] = "humanevalfix/Python/0"
    line["version"] = "fixed"
    line["state"] = "proven"
    scan = tmp_path / "scan.jsonl"
    scan.write_text(json.dumps(line) + "\n", encoding="utf-8")
    (entry,) = viewer.load_entries(scan)
    assert entry.project == "demo-shop"
    assert entry.result.state is State.PROVEN


def test_blank_lines_are_skipped_but_counted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    scan = tmp_path / "scan.jsonl"
    scan.write_text("\n   \n" + first + "\n" + "bogus\n", encoding="utf-8")
    assert main(["view", str(scan)]) == 2
    assert "line 4" in capsys.readouterr().err


def test_malformed_line_exits_2_and_names_line_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    first = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    scan = tmp_path / "scan.jsonl"
    scan.write_text(first + "\n" + "not json\n", encoding="utf-8")
    assert main(["view", str(scan)]) == 2
    assert "line 2" in capsys.readouterr().err
    assert opened == []


def test_non_string_project_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    scan = tmp_path / "scan.jsonl"
    scan.write_text('{"project": 7, "result": {}}\n', encoding="utf-8")
    assert main(["view", str(scan)]) == 2
    assert "line 1" in capsys.readouterr().err
    assert opened == []


def test_missing_file_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    assert main(["view", str(tmp_path / "missing.jsonl")]) == 2
    assert "missing.jsonl" in capsys.readouterr().err
    assert opened == []


def test_port_in_use_exits_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        port = str(holder.getsockname()[1])
        assert main(["view", str(FIXTURE), "--port", port]) == 2
    finally:
        holder.close()
    assert port in capsys.readouterr().err
    assert opened == []


def test_help_lists_view_and_port(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "view" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc_view:
        main(["view", "--help"])
    assert exc_view.value.code == 0
    assert "--port" in capsys.readouterr().out


def test_view_uses_no_model_sandbox_or_pipeline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``view`` serves the page while the model, sandbox and pipeline doors raise."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("view must not call models, the sandbox or the pipeline")

    monkeypatch.setattr(models_module, "build_client", _boom)
    monkeypatch.setattr(pipeline_module, "run", _boom)
    monkeypatch.setattr(sandbox_run_module, "run_python", _boom)
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    captured: dict[str, ThreadingHTTPServer] = {}
    real_create_server = viewer.create_server

    def _spy(entries: tuple[viewer.Entry, ...], port: int) -> ThreadingHTTPServer:
        server = real_create_server(entries, port)
        captured["server"] = server
        return server

    monkeypatch.setattr(viewer, "create_server", _spy)
    codes: list[int] = []
    thread = threading.Thread(target=lambda: codes.append(main(["view", str(FIXTURE), "--port", "0"])), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10.0
        while "server" not in captured:
            assert time.monotonic() < deadline, "view server did not start"
            time.sleep(0.05)
        server = captured["server"]
        base = _base(server)
        _wait_ready(base)
        status, _, body = _get(base + "/api/results")
        assert status == 200
        payload = json.loads(body.decode("utf-8"))
        assert [entry["result"]["state"] for entry in payload] == ["proven", "counterexample", "open", "not_checkable"]
        assert opened == [base + "/"]
        assert "127.0.0.1" in capsys.readouterr().out
    finally:
        captured["server"].shutdown()
        thread.join(timeout=10)
    assert codes == [0]


def test_viewer_html_ships_in_the_wheel() -> None:
    """``viewer.html`` is installed with the wheel, so ``page_html`` works outside the source tree.

    The wheel target ships whole package directories (``packages = ["src/pcc"]``), so the page has to
    live inside one of them; otherwise an installed ``pcc-check view`` would not find its page. Any
    file-selection override must keep covering ``viewer.html``. A full wheel build for this issue
    (hatchling 1.33.0, 87 files) was listed once and contains ``pcc/report/viewer.html``.
    """
    assert viewer.page_html().strip()
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    build = data["tool"]["hatch"]["build"]
    wheel_target = build["targets"]["wheel"]
    packages = wheel_target["packages"]
    html = Path("src/pcc/report/viewer.html")
    assert (repo_root / html).is_file()
    assert any(str(html).startswith(pkg.rstrip("/") + "/") for pkg in packages)
    html_posix = html.as_posix()
    for section in (build, wheel_target):
        for pattern in section.get("exclude", []):
            assert not fnmatch.fnmatch(html_posix, pattern), f"hatch exclude {pattern!r} drops {html_posix}"
        only = section.get("only-include")
        if only is not None:
            assert any(fnmatch.fnmatch(html_posix, pattern) for pattern in only), (
                f"hatch only-include {only!r} drops {html_posix}"
            )
