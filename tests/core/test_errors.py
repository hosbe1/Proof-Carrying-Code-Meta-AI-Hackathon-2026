from __future__ import annotations

import pickle

import pytest

from pcc.core import PccError, ToolError, UnsupportedConstruct


def test_tool_error_is_not_an_expected_failure() -> None:
    assert not issubclass(ToolError, PccError)
    assert issubclass(ToolError, Exception)


def test_unsupported_construct_message_is_not_doubled() -> None:
    assert str(UnsupportedConstruct("match statement")) == "unsupported construct: match statement"
    assert str(UnsupportedConstruct("match statement", 7)) == "unsupported construct: match statement (line 7)"


@pytest.mark.parametrize("lineno", [None, 7])
def test_unsupported_construct_survives_pickle(lineno: int | None) -> None:
    error = UnsupportedConstruct("match statement", lineno)
    copy = pickle.loads(pickle.dumps(error))
    assert str(copy) == str(error)
    assert (copy.construct, copy.lineno) == ("match statement", lineno)
    assert copy.args == error.args
