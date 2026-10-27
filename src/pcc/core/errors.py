"""Exceptions that cross step boundaries."""


class PccError(Exception):
    """Base class for expected failures that the pipeline turns into a result state."""


class UnsupportedConstruct(PccError):
    """The transpiler does not support this construct yet; the pipeline falls back to AI translation."""

    def __init__(self, construct: str, lineno: int | None = None) -> None:
        # ``args`` holds the constructor arguments, not the message, so that pickle rebuilds the error from them.
        super().__init__(construct, lineno)
        self.construct = construct
        self.lineno = lineno

    def __str__(self) -> str:
        where = f" (line {self.lineno})" if self.lineno is not None else ""
        return f"unsupported construct: {self.construct}{where}"


class NotCheckable(PccError):
    """Intent, spec or translation failed its checks; the function ends as ``not_checkable`` with this reason."""


class ToolError(Exception):
    """A tool the pipeline needs is broken or unreachable: dafny missing or of the wrong version, a model API down.

    This says nothing about the code under test, so it is not an expected failure and never becomes a result state.
    It must escape the pipeline as an exception.
    """
