"""Unit tests for ``pcc.verifier.errors.parse_errors``.

The inputs are recorded outputs captured from real dafny 4.11 runs, so no Dafny CLI is needed here:

- ``postcondition.txt``: ``dafny verify --resource-limit 1000000 --verification-time-limit 0 post.dfy``
  on a method whose postcondition fails.
- ``pylib_error.txt``: ``dafny resolve main.dfy PyLib.dfy`` where the ``PyLib.dfy`` copy is the real
  ``PyLib.dfy`` with one broken lemma appended.
- ``windows_crlf.txt``: ``dafny resolve`` on a file named ``C:\\proj\\main.dfy``. Dotnet normalizes
  backslashes to forward slashes in diagnostic lines on Linux, so the separators were restored to
  backslashes as dafny on Windows emits them (the summary line already used backslashes); the line
  endings are CRLF as on Windows.
"""

from __future__ import annotations

from pathlib import Path

from pcc.verifier.errors import parse_errors

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def read_output(name: str) -> str:
    return (OUTPUTS_DIR / name).read_text(encoding="utf-8")


def test_postcondition_error_carries_related_with_source() -> None:
    errors = parse_errors(read_output("postcondition.txt"))
    assert len(errors) == 1
    error = errors[0]
    assert (error.line, error.column) == (3, 0)
    assert error.message == "a postcondition could not be proved on this return path"
    assert error.related == ("post.dfy(2,12): this is the postcondition that could not be proved\n  ensures r > x")


def test_pylib_errors_carry_pylib_prefix() -> None:
    errors = parse_errors(read_output("pylib_error.txt"))
    assert len(errors) == 2
    assert (errors[0].line, errors[0].column) == (81, 13)
    for error in errors:
        assert error.message.startswith("PyLib.dfy: ")
    assert "unresolved identifier: UndefinedName" in errors[0].message


def test_windows_path_and_crlf_output_parses() -> None:
    errors = parse_errors(read_output("windows_crlf.txt"))
    assert len(errors) == 1
    error = errors[0]
    assert (error.line, error.column) == (2, 11)
    assert error.message == "unresolved identifier: UndefinedName"
    assert error.related == ""


def test_empty_output_gives_no_errors() -> None:
    assert parse_errors("") == ()
    assert parse_errors("\nDafny program verifier did not attempt verification\n") == ()


def test_related_location_without_error_is_ignored() -> None:
    output = "post.dfy(2,12): Related location: this is the postcondition that could not be proved\n"
    assert parse_errors(output) == ()


def test_each_error_keeps_its_own_related() -> None:
    output = (
        "a.dfy(3,0): Error: first failure\n"
        "  |\n"
        "3 | {\n"
        "  | ^\n"
        "\n"
        "a.dfy(2,5): Related location: first cause\n"
        "  |\n"
        "2 |   ensures x > 0\n"
        "  |           ^\n"
        "\n"
        "b.dfy(7,1): Error: second failure\n"
        "  |\n"
        "7 |   var y := z;\n"
        "  |            ^\n"
    )
    errors = parse_errors(output)
    assert len(errors) == 2
    assert errors[0].related == "a.dfy(2,5): first cause\n  ensures x > 0"
    assert errors[1].message == "second failure"
    assert errors[1].related == ""
