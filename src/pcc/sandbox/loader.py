"""Binding the function under test to the name ``target`` in the sandbox."""

from __future__ import annotations

from pcc.core import FunctionRef


def target_loader(function: FunctionRef) -> str:
    """Python source that binds ``function`` to the name ``target``.

    The module path comes from the file path with a leading ``src/`` dropped (``src/cart.py`` becomes ``cart``,
    ``strings/palindrome.py`` becomes ``strings.palindrome``). The generated source imports the module, then walks
    the parts of the qualname (with every ``<locals>.`` removed): while the parent is a class it uses attribute
    access. When attribute access fails, the parent is a function (a nested function or a named lambda), so it
    ``exec``s the function's own source in a copy of the module's globals and takes the last part of the qualname.
    """
    module = _module_name(function.path)
    parts = [part for part in function.qualname.split(".") if part != "<locals>"]
    lines = [
        "import importlib",
        f"_module = importlib.import_module({module!r})",
        f"_parts = {parts!r}",
        f"_source = {function.source!r}",
        "_target = _module",
        "try:",
        "    for _part in _parts:",
        "        _target = getattr(_target, _part)",
        "    target = _target",
        "except AttributeError:",
        "    _namespace = dict(vars(_module))",
        "    exec(compile(_source, '<target>', 'exec'), _namespace)",
        "    target = _namespace[_parts[-1]]",
    ]
    return "\n".join(lines) + "\n"


def _module_name(path: str) -> str:
    """The dotted module path for ``path`` (POSIX separators, a leading ``src/`` dropped)."""
    relative = path.removeprefix("src/")
    if relative.endswith(".py"):
        relative = relative[: -len(".py")]
    return relative.replace("/", ".")


__all__ = ["target_loader"]
