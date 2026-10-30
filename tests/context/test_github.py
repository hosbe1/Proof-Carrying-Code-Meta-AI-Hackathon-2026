"""Tests for GitHub PR/issue evidence (issue #24).

Every test that talks to the API uses a localhost stub; the base URL is
injectable through the ``GITHUB_API_URL`` env key so no real network is used.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from pcc.context import collect
from pcc.context.github import github_evidence
from pcc.core import Evidence, FunctionRef, SourceKind, source_hash

MAX_EXCERPT = 2000


def _ref(source: str, path: str = "src/m.py", qualname: str = "f", lineno: int = 1) -> FunctionRef:
    return FunctionRef(path=path, qualname=qualname, lineno=lineno, source=source, source_hash=source_hash(source))


def _commit_evidence(excerpt: str, ref: str = "commit abc1234") -> Evidence:
    return Evidence(SourceKind.COMMIT, ref, excerpt)


class _StubState:
    def __init__(self) -> None:
        self.pulls: dict[int, dict[str, Any]] = {}
        self.issues: dict[int, dict[str, Any]] = {}
        self.hits: list[str] = []


def _serve(state: _StubState) -> tuple[ThreadingHTTPServer, threading.Thread]:
    parent: dict[str, Any] = {"state": state}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            st: _StubState = parent["state"]
            st.hits.append(self.path)
            if self.path.startswith("/repos/o/r/pulls/"):
                try:
                    number = int(self.path.rsplit("/", 1)[-1].split("?", 1)[0])
                except ValueError:
                    number = -1
                payload = st.pulls.get(number)
                if payload is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/repos/o/r/issues/"):
                try:
                    number = int(self.path.rsplit("/", 1)[-1].split("?", 1)[0])
                except ValueError:
                    number = -1
                payload = st.issues.get(number)
                if payload is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _env(
    server: ThreadingHTTPServer, tmp_path: Path, pr_number: int | None = 7, *, token: str = "token"
) -> dict[str, str]:
    port = server.server_address[1]
    env: dict[str, str] = {
        "GITHUB_TOKEN": token,
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_API_URL": f"http://127.0.0.1:{port}",
    }
    if pr_number is not None:
        event = tmp_path / "event.json"
        event.write_text(json.dumps({"pull_request": {"number": pr_number}}), encoding="utf-8")
        env["GITHUB_EVENT_PATH"] = str(event)
    return env


def test_without_token_returns_empty() -> None:
    assert github_evidence([_commit_evidence("Fixes #12")], {}) == ()
    assert github_evidence([_commit_evidence("Fixes #12")], {"GITHUB_REPOSITORY": "o/r"}) == ()


def test_without_repository_returns_empty() -> None:
    assert github_evidence([_commit_evidence("Fixes #12")], {"GITHUB_TOKEN": "t"}) == ()


def test_pr_body_fix_gives_pr_and_issue(tmp_path: Path) -> None:
    state = _StubState()
    state.pulls[7] = {"title": "Fix total", "body": "Fixes #12"}
    state.issues[12] = {"title": "Wrong total", "body": "total is off"}
    server, _ = _serve(state)
    try:
        env = _env(server, tmp_path)
        result = github_evidence([], env)
    finally:
        server.shutdown()
    kinds = [item.kind for item in result]
    assert kinds == [SourceKind.PULL_REQUEST, SourceKind.ISSUE]
    assert result[0].ref == "#7"
    assert "Fix total" in result[0].excerpt
    assert result[1].ref == "#12"
    assert "Wrong total" in result[1].excerpt


def test_404_issue_is_skipped(tmp_path: Path) -> None:
    state = _StubState()
    state.pulls[7] = {"title": "Fix all", "body": "Fixes #12 and #13"}
    state.issues[12] = {"title": "First", "body": "first body"}
    server, _ = _serve(state)
    try:
        env = _env(server, tmp_path)
        result = github_evidence([], env)
    finally:
        server.shutdown()
    refs = [item.ref for item in result if item.kind is SourceKind.ISSUE]
    assert refs == ["#12"]


def test_code_block_ref_still_counts(tmp_path: Path) -> None:
    state = _StubState()
    state.pulls[7] = {"title": "Docs", "body": "Example:\n```\nvalue = f(#99)\n```\nSee #12"}
    state.issues[99] = {"title": "Ninety nine", "body": "body 99"}
    state.issues[12] = {"title": "Twelve", "body": "body 12"}
    server, _ = _serve(state)
    try:
        env = _env(server, tmp_path)
        result = github_evidence([], env)
    finally:
        server.shutdown()
    refs = [item.ref for item in result if item.kind is SourceKind.ISSUE]
    assert refs == ["#99", "#12"]


def test_duplicate_numbers_fetched_once(tmp_path: Path) -> None:
    state = _StubState()
    state.pulls[7] = {"title": "Fix", "body": "Fixes #12"}
    state.issues[12] = {"title": "Twelve", "body": "body"}
    server, _ = _serve(state)
    try:
        env = _env(server, tmp_path)
        commits = [_commit_evidence("Retry fix for #12"), _commit_evidence("Again #12")]
        result = github_evidence(commits, env)
    finally:
        server.shutdown()
    refs = [item.ref for item in result if item.kind is SourceKind.ISSUE]
    assert refs == ["#12"]
    assert state.hits.count("/repos/o/r/issues/12") == 1


def test_at_most_five_issues(tmp_path: Path) -> None:
    state = _StubState()
    state.pulls[7] = {"title": "Big", "body": " ".join(f"#{n}" for n in range(1, 8))}
    for n in range(1, 8):
        state.issues[n] = {"title": f"Issue {n}", "body": f"body {n}"}
    server, _ = _serve(state)
    try:
        env = _env(server, tmp_path)
        result = github_evidence([], env)
    finally:
        server.shutdown()
    refs = [item.ref for item in result if item.kind is SourceKind.ISSUE]
    assert refs == ["#1", "#2", "#3", "#4", "#5"]


def test_commit_messages_are_mined_first_seen_order(tmp_path: Path) -> None:
    state = _StubState()
    for n in (21, 22):
        state.issues[n] = {"title": f"Issue {n}", "body": f"body {n}"}
    server, _ = _serve(state)
    try:
        env = _env(server, tmp_path, pr_number=None)
        commits = [_commit_evidence("Fix #22 first"), _commit_evidence("Also #21 and #22")]
        result = github_evidence(commits, env)
    finally:
        server.shutdown()
    refs = [item.ref for item in result if item.kind is SourceKind.ISSUE]
    assert refs == ["#22", "#21"]
    assert all(item.kind is not SourceKind.PULL_REQUEST for item in result)


def test_excerpts_are_cut_to_2000_chars(tmp_path: Path) -> None:
    state = _StubState()
    state.pulls[7] = {"title": "T", "body": "y" * 2500}
    state.issues[12] = {"title": "I", "body": "z" * 2500}
    server, _ = _serve(state)
    try:
        env = _env(server, tmp_path)
        result = github_evidence([], env)
    finally:
        server.shutdown()
    for item in result:
        assert len(item.excerpt) == MAX_EXCERPT + 1
        assert item.excerpt.endswith("…")


def test_network_error_warns_and_returns_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    state = _StubState()
    server, _ = _serve(state)
    port = server.server_address[1]
    server.shutdown()
    server.server_close()
    env = {
        "GITHUB_TOKEN": "t",
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_API_URL": f"http://127.0.0.1:{port}",
        "GITHUB_EVENT_PATH": str(tmp_path / "missing.json"),
    }
    with caplog.at_level(logging.WARNING, logger="pcc.context.github"):
        result = github_evidence([_commit_evidence("Fixes #12")], env)
    assert result == ()
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(repo: Path, *args: str) -> None:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False, env=dict(os.environ))
    assert proc.returncode == 0, proc.stderr


def test_collect_passes_environ_and_empty_env_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "def f() -> int:\n    return 1\n"
    _write(repo, "src/m.py", source)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", ".")
    _git(repo, "commit", "--no-verify", "-m", "init")
    function = _ref(source)
    state = _StubState()
    state.pulls[7] = {"title": "Fix f", "body": "Fixes #12"}
    state.issues[12] = {"title": "Broken f", "body": "f is broken"}
    server, _ = _serve(state)
    try:
        port = server.server_address[1]
        event = tmp_path / "event.json"
        event.write_text(json.dumps({"pull_request": {"number": 7}}), encoding="utf-8")
        full = {
            "GITHUB_TOKEN": "t",
            "GITHUB_REPOSITORY": "o/r",
            "GITHUB_API_URL": f"http://127.0.0.1:{port}",
            "GITHUB_EVENT_PATH": str(event),
        }
        for key, value in full.items():
            monkeypatch.setenv(key, value)
        with_env = collect(function, repo)
        assert SourceKind.PULL_REQUEST in [item.kind for item in with_env.evidence]
        assert "#12" in [item.ref for item in with_env.evidence if item.kind is SourceKind.ISSUE]
        for key in full:
            monkeypatch.delenv(key, raising=False)
        empty = collect(function, repo)
        kinds = [item.kind for item in empty.evidence]
        assert SourceKind.PULL_REQUEST not in kinds
        assert SourceKind.ISSUE not in kinds
    finally:
        server.shutdown()
