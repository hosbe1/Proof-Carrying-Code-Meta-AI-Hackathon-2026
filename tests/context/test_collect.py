"""Tests for the fixed context collector (docs/plan.md, step 1; issue #13).

Every test builds its own repository under ``tmp_path``; the ones that need history run ``git init`` there.
Fixture commits use ``--no-verify`` so that environment-level commit hooks (for example an author-identity
hook) cannot reject them; where the hooks do not exist this flag is a no-op.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest

from pcc.context import collect
from pcc.core import Context, FunctionRef, SourceKind, source_hash

FIXED_DATES = {
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}

MAX_EXCERPT = 2000


def _write(repo: Path, rel: str, text: str) -> Path:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env is not None:
        merged.update(env)
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False, env=merged)


def _init_repo(repo: Path) -> None:
    proc = _git(repo, "init", "-b", "main")
    assert proc.returncode == 0, proc.stderr
    assert _git(repo, "config", "user.email", "collector-test@example.com").returncode == 0
    assert _git(repo, "config", "user.name", "Collector Test").returncode == 0


def _commit(repo: Path, subject: str, body: str = "", *, dates: bool = False) -> None:
    args = ["commit", "--no-verify", "-m", subject]
    if body:
        args += ["-m", body]
    proc = _git(repo, *args, env=FIXED_DATES if dates else None)
    assert proc.returncode == 0, proc.stderr


def _ref(source: str, path: str, qualname: str, lineno: int) -> FunctionRef:
    return FunctionRef(
        path=path, qualname=qualname, lineno=lineno, source=source, source_hash=source_hash(source)
    )


def _kinds(context: Context) -> list[SourceKind]:
    return [item.kind for item in context.evidence]


FUNC_SOURCE = (
    "def total(xs: list[int], discount: float = 0.0) -> float:\n"
    '    """Return the discounted total.\n'
    "\n"
    "    >>> total([10, 20])\n"
    "    30.0\n"
    '    """\n'
    "    # bug: off by one\n"
    "    result = sum(xs)\n"
    "    return result - discount\n"
)

CART_SOURCE = "# cart helpers\n# totals include tax\n" + FUNC_SOURCE


def _rich_repo(repo: Path) -> FunctionRef:
    """A repository with a docstring, comments, a test, a call site, history and docs."""
    _init_repo(repo)
    _write(repo, "src/cart.py", CART_SOURCE)
    _write(
        repo,
        "src/shop.py",
        "from cart import total\n"
        "\n"
        "VALUE = 10\n"
        "FREE_SHIPPING = 50\n"
        "price = total([VALUE, FREE_SHIPPING])\n"
        "print(price)\n"
        "done = True\n",
    )
    _write(
        repo,
        "tests/test_cart.py",
        "from cart import total\n"
        "\n"
        "\n"
        "def test_total():\n"
        "    assert total([10, 20]) == 30.0\n"
        "\n"
        "\n"
        "def helper_not_a_test():\n"
        "    return total([1])\n",
    )
    _write(
        repo,
        "README.md",
        "# Cart\n"
        "\n"
        "The total helper adds up prices.\n"
        "\n"
        "Unrelated paragraph.\n",
    )
    _write(
        repo,
        "docs/guide.md",
        "# Guide\n"
        "\n"
        "Call total after discounts.\n"
        "\n"
        "The total never includes shipping.\n"
        "\n"
        "Another total note for the limit.\n"
        "\n"
        "One more total paragraph past the limit.\n",
    )
    _git(repo, "add", ".")
    _commit(repo, "add cart total", "First version of the cart helper.")
    return _ref(FUNC_SOURCE, "src/cart.py", "total", 3)


def test_collect_docstring_and_test(tmp_path: Path) -> None:
    """Example 1: a function with a docstring and a test."""
    repo = tmp_path
    function = _rich_repo(repo)
    context = collect(function, repo)
    kinds = _kinds(context)
    assert SourceKind.NAME in kinds
    assert SourceKind.SIGNATURE in kinds
    assert SourceKind.DOCSTRING in kinds
    assert SourceKind.TEST in kinds
    docstrings = [item for item in context.evidence if item.kind is SourceKind.DOCSTRING]
    assert len(docstrings) == 1
    assert "Return the discounted total." in docstrings[0].excerpt
    assert ">>> total([10, 20])" in docstrings[0].excerpt
    tests = [item for item in context.evidence if item.kind is SourceKind.TEST]
    assert [item.ref for item in tests] == ["tests/test_cart.py::test_total"]
    assert "assert total([10, 20]) == 30.0" in tests[0].excerpt


def test_collect_without_git(tmp_path: Path) -> None:
    """Example 2: a repository without git gives no COMMIT evidence and no error."""
    repo = tmp_path
    _write(repo, "src/cart.py", CART_SOURCE)
    function = _ref(FUNC_SOURCE, "src/cart.py", "total", 3)
    context = collect(function, repo)
    assert [item for item in context.evidence if item.kind is SourceKind.COMMIT] == []
    assert SourceKind.NAME in _kinds(context)


def test_collect_body_comment(tmp_path: Path) -> None:
    """Example 3: a body comment lands in COMMENT evidence without leaking the source."""
    repo = tmp_path
    _write(repo, "src/cart.py", CART_SOURCE)
    function = _ref(FUNC_SOURCE, "src/cart.py", "total", 3)
    context = collect(function, repo)
    comments = [item for item in context.evidence if item.kind is SourceKind.COMMENT]
    assert "bug: off by one" in [item.excerpt for item in comments]
    for item in context.evidence:
        assert function.source not in item.excerpt


def test_common_word_matches_identifier_not_text(tmp_path: Path) -> None:
    """Edge case: a common word such as ``max`` only matches as an identifier (ast), not as text."""
    repo = tmp_path
    source = "def max(values: list[int]) -> int:\n    m = values[0]\n    return m\n"
    _write(repo, "src/m.py", source)
    _write(
        repo,
        "tests/test_m.py",
        "def test_maximum():\n    maximum = 5\n    assert maximum == 5\n",
    )
    _write(repo, "src/use.py", "# maximize everything\nlabel = 'maximize'\n")
    function = _ref(source, "src/m.py", "max", 1)
    context = collect(function, repo)
    assert [item for item in context.evidence if item.kind is SourceKind.TEST] == []
    assert [item for item in context.evidence if item.kind is SourceKind.CALL_SITE] == []


def test_common_word_identifier_match_is_found(tmp_path: Path) -> None:
    """The same common word used as a real identifier is collected."""
    repo = tmp_path
    source = "def max(values: list[int]) -> int:\n    m = values[0]\n    return m\n"
    _write(repo, "src/m.py", source)
    _write(repo, "tests/test_m.py", "def test_max():\n    assert max([3, 1]) == 3\n")
    function = _ref(source, "src/m.py", "max", 1)
    context = collect(function, repo)
    tests = [item for item in context.evidence if item.kind is SourceKind.TEST]
    assert [item.ref for item in tests] == ["tests/test_m.py::test_max"]


def test_own_lines_never_count_as_call_site(tmp_path: Path) -> None:
    """Edge case: a recursive call inside the function is not a call site; outside calls are."""
    repo = tmp_path
    source = "def fact(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n"
    _write(repo, "src/m.py", source + "\n\nvalue = fact(5)\n")
    function = _ref(source, "src/m.py", "fact", 1)
    context = collect(function, repo)
    calls = [item for item in context.evidence if item.kind is SourceKind.CALL_SITE]
    assert len(calls) == 1
    assert calls[0].ref == "src/m.py:7"


def test_new_uncommitted_lines_have_no_commit_history(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Edge case: new, uncommitted lines have no COMMIT history; a warning is logged, never raised."""
    repo = tmp_path
    _init_repo(repo)
    _write(repo, "src/a.py", "x = 1\n")
    _git(repo, "add", ".")
    _commit(repo, "init")
    new_source = "def fresh() -> int:\n    return 1\n"
    _write(repo, "src/a.py", "x = 1\n" + new_source)
    function = _ref(new_source, "src/a.py", "fresh", 2)
    with caplog.at_level(logging.WARNING, logger="pcc.context.collect"):
        context = collect(function, repo)
    assert [item for item in context.evidence if item.kind is SourceKind.COMMIT] == []
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_source_never_appears_in_any_excerpt(tmp_path: Path) -> None:
    """The intent step never sees the body, so no excerpt may contain the whole source."""
    repo = tmp_path
    function = _rich_repo(repo)
    context = collect(function, repo)
    assert len(context.evidence) > 0
    for item in context.evidence:
        assert function.source not in item.excerpt, (item.kind, item.ref)


def test_evidence_order_and_refs(tmp_path: Path) -> None:
    """Evidence comes in ticket order with the documented ref formats."""
    repo = tmp_path
    function = _rich_repo(repo)
    context = collect(function, repo)
    assert [(item.kind, item.ref) for item in context.evidence] == [
        (SourceKind.NAME, "src/cart.py:3"),
        (SourceKind.SIGNATURE, "src/cart.py:3"),
        (SourceKind.DOCSTRING, "src/cart.py:3"),
        (SourceKind.COMMENT, "src/cart.py:1"),
        (SourceKind.COMMENT, "src/cart.py:9"),
        (SourceKind.TEST, "tests/test_cart.py::test_total"),
        (SourceKind.CALL_SITE, "src/shop.py:5"),
        (SourceKind.COMMIT, context.evidence[7].ref),
        (SourceKind.DOCS, "README.md"),
        (SourceKind.DOCS, "docs/guide.md"),
        (SourceKind.DOCS, "docs/guide.md"),
    ]
    assert context.evidence[7].ref.startswith("commit ")
    assert "add cart total" in context.evidence[7].excerpt
    assert "First version of the cart helper." in context.evidence[7].excerpt
    call = context.evidence[6]
    assert call.excerpt == (
        "VALUE = 10\nFREE_SHIPPING = 50\nprice = total([VALUE, FREE_SHIPPING])\nprint(price)\ndone = True"
    )


def test_git_log_L_exact_output_on_two_commit_repo(tmp_path: Path) -> None:
    """Acceptance: pin the exact output of the ticket's ``git log -L`` command."""
    repo = tmp_path
    _init_repo(repo)
    _write(repo, "src/a.py", "def f(x: int) -> int:\n    return x + 1\n")
    _git(repo, "add", ".")
    _commit(repo, "first subject", "first body", dates=True)
    _write(repo, "src/a.py", "def f(x: int) -> int:\n    return x + 2\n")
    _git(repo, "add", ".")
    _commit(repo, "second subject", "second body", dates=True)
    proc = _git(repo, "log", "-L", "1,2:src/a.py", "--format=%h%n%s%n%b", "--no-patch")
    assert proc.returncode == 0
    new_hash = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    old_hash = _git(repo, "rev-parse", "--short", "HEAD~1").stdout.strip()
    assert proc.stdout == (
        f"{new_hash}\nsecond subject\nsecond body\n\n{old_hash}\nfirst subject\nfirst body\n\n"
    )
    assert not any(line.startswith("diff ") for line in proc.stdout.splitlines())


def test_comment_block_above_def_is_one_evidence(tmp_path: Path) -> None:
    repo = tmp_path
    function = _rich_repo(repo)
    context = collect(function, repo)
    block = context.evidence[3]
    assert block.kind is SourceKind.COMMENT
    assert block.excerpt == "cart helpers\ntotals include tax"


def test_multiline_signature(tmp_path: Path) -> None:
    repo = tmp_path
    source = (
        "def combine(\n    first: str,\n    second: str = 'x',\n) -> str:\n    return first + second\n"
    )
    _write(repo, "src/m.py", source)
    function = _ref(source, "src/m.py", "combine", 1)
    context = collect(function, repo)
    signatures = [item for item in context.evidence if item.kind is SourceKind.SIGNATURE]
    assert len(signatures) == 1
    assert signatures[0].excerpt == (
        "def combine(\n    first: str,\n    second: str = 'x',\n) -> str:"
    )


def test_limits_are_enforced(tmp_path: Path) -> None:
    repo = tmp_path
    lines = ["def f() -> int:"]
    for number in range(12):
        lines.append(f"    # note {number}")
    lines.append("    return 1")
    source = "\n".join(lines) + "\n"
    _write(repo, "src/m.py", source)
    test_file = "\n\n".join(
        f"def test_{number}():\n    assert f() == 1" for number in range(7)
    )
    _write(repo, "tests/test_m.py", test_file + "\n")
    function = _ref(source, "src/m.py", "f", 1)
    context = collect(function, repo)
    assert len([item for item in context.evidence if item.kind is SourceKind.COMMENT]) == 10
    assert len([item for item in context.evidence if item.kind is SourceKind.TEST]) == 5


def test_long_test_function_is_capped_at_40_lines(tmp_path: Path) -> None:
    repo = tmp_path
    source = "def f() -> int:\n    return 1\n"
    _write(repo, "src/m.py", source)
    body = "\n".join(f"    assert f() == 1  # line {number}" for number in range(60))
    _write(repo, "tests/test_m.py", f"def test_long():\n{body}\n")
    function = _ref(source, "src/m.py", "f", 1)
    context = collect(function, repo)
    tests = [item for item in context.evidence if item.kind is SourceKind.TEST]
    assert len(tests) == 1
    assert len(tests[0].excerpt.splitlines()) == 40


def test_long_excerpt_is_cut_off(tmp_path: Path) -> None:
    repo = tmp_path
    doc = "y" * 2500
    source = f'def f() -> int:\n    """{doc}"""\n    return 1\n'
    _write(repo, "src/m.py", source)
    function = _ref(source, "src/m.py", "f", 1)
    context = collect(function, repo)
    docstrings = [item for item in context.evidence if item.kind is SourceKind.DOCSTRING]
    assert len(docstrings) == 1
    assert docstrings[0].excerpt == "y" * MAX_EXCERPT + "…"
