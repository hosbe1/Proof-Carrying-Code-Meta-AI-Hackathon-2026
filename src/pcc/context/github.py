"""Pull request and linked-issue evidence from the GitHub API (docs/plan.md, step 1).

The pull request description and the linked issues come from the GitHub API with the ``GITHUB_TOKEN`` of the workflow
run. Without a token they are skipped. Called by :func:`pcc.context.collect.collect` with the commit evidence and
``os.environ``; the API base URL is injectable through ``GITHUB_API_URL`` for tests (default
``https://api.github.com``). Optional sources never raise: network errors log a warning and yield nothing.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from pcc.core import Evidence, SourceKind

logger = logging.getLogger(__name__)

MAX_ISSUES = 5
MAX_EXCERPT_CHARS = 2000
_REQUEST_TIMEOUT_SECONDS = 10.0
_DEFAULT_API_URL = "https://api.github.com"

_ISSUE_REF_RE = re.compile(r"#(\d+)")

__all__ = ["github_evidence"]


def github_evidence(commits: Sequence[Evidence], env: Mapping[str, str]) -> tuple[Evidence, ...]:
    """Pull request evidence plus issue evidence for every ``#N`` in the PR body and commit messages.

    Needs ``GITHUB_TOKEN`` and ``GITHUB_REPOSITORY``; the pull request number comes from the JSON in
    ``GITHUB_EVENT_PATH`` (``pull_request.number``). Returns ``()`` without a token or repository, on any network
    error (after a warning), and never raises for these optional sources. A 404 for a single issue skips that issue.
    At most five distinct issues are fetched, in first-seen order (PR body first, then commits in given order); every
    excerpt is cut to 2,000 characters.
    """
    token = (env.get("GITHUB_TOKEN") or "").strip()
    repository = (env.get("GITHUB_REPOSITORY") or "").strip()
    if not token or not repository:
        return ()
    base_url = (env.get("GITHUB_API_URL") or _DEFAULT_API_URL).rstrip("/") or _DEFAULT_API_URL
    pr_number = _pr_number(env)
    pr_body = ""
    found: list[Evidence] = []
    if pr_number is not None:
        try:
            pr_data = _get_json(f"{base_url}/repos/{repository}/pulls/{pr_number}", token)
        except _HardFetchError as error:
            logger.warning("GitHub pull request #%d for %s failed: %s", pr_number, repository, error)
            return ()
        if pr_data is not None:
            title = _as_text(pr_data.get("title"))
            pr_body = _as_text(pr_data.get("body"))
            found.append(Evidence(SourceKind.PULL_REQUEST, f"#{pr_number}", _cut(_join_title_body(title, pr_body))))
    numbers = _issue_numbers(pr_body, commits)
    for number in numbers:
        try:
            issue_data = _get_json(f"{base_url}/repos/{repository}/issues/{number}", token)
        except _HardFetchError as error:
            logger.warning("GitHub issue #%d for %s failed: %s", number, repository, error)
            return ()
        if issue_data is None:
            continue
        title = _as_text(issue_data.get("title"))
        body = _as_text(issue_data.get("body"))
        found.append(Evidence(SourceKind.ISSUE, f"#{number}", _cut(_join_title_body(title, body))))
    return tuple(found)


class _HardFetchError(Exception):
    """A fetch failure that aborts all GitHub evidence (network error, non-404 HTTP status, bad payload)."""


def _pr_number(env: Mapping[str, str]) -> int | None:
    """The pull request number from ``GITHUB_EVENT_PATH``, or None when it is missing or unusable."""
    event_path = (env.get("GITHUB_EVENT_PATH") or "").strip()
    if not event_path:
        return None
    try:
        with open(event_path, encoding="utf-8") as handle:
            payload: Any = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        logger.warning("GitHub event file %s cannot be read: %s", event_path, error)
        return None
    if not isinstance(payload, dict):
        return None
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        return None
    number = pull_request.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        return None
    return number


def _issue_numbers(pr_body: str, commits: Sequence[Evidence]) -> list[int]:
    """Distinct ``#N`` numbers, first-seen order: PR body first, then commits; at most five."""
    seen: list[int] = []
    for text in [pr_body, *[commit.excerpt for commit in commits]]:
        for match in _ISSUE_REF_RE.finditer(text):
            try:
                number = int(match.group(1))
            except ValueError:
                continue
            if number <= 0 or number in seen:
                continue
            seen.append(number)
            if len(seen) >= MAX_ISSUES:
                return seen
    return seen


def _get_json(url: str, token: str) -> dict[str, Any] | None:
    """GET ``url`` as JSON. None on 404; raises :class:`_HardFetchError` otherwise unusable."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "pcc-context",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise _HardFetchError(f"GET {url} returned HTTP {error.code}") from error
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise _HardFetchError(f"GET {url} failed: {error}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _HardFetchError(f"GET {url} returned invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise _HardFetchError(f"GET {url} returned a JSON non-object")
    return payload


def _as_text(value: Any) -> str:
    """The string value of a JSON field, or empty when it is missing or null."""
    return value if isinstance(value, str) else ""


def _join_title_body(title: str, body: str) -> str:
    """Title plus body in the commit-excerpt shape (subject, blank line, body)."""
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def _cut(text: str) -> str:
    """Truncate excerpts longer than 2,000 characters with an ellipsis."""
    if len(text) <= MAX_EXCERPT_CHARS:
        return text
    return text[:MAX_EXCERPT_CHARS] + "…"
