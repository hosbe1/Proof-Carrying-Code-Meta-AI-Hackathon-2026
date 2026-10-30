"""Parameter and return types inferred with pyright (docs/plan.md, step 4, stretch goal).

Functions without type hints are ``not_checkable`` ("no types") unless pyright can
infer their types. ``infer_types`` reports those types so the transpiler can use them;
anything it cannot infer stays ``None``.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pcc.core import FunctionRef

logger = logging.getLogger(__name__)

#: Bound on ``pyright --outputjson``: a stuck pyright process means no types, not a hang.
_PYRIGHT_TIMEOUT_SECONDS = 120.0

#: ``Type of "f" is "(x: int) -> int"``: the diagnostic a ``reveal_type`` call produces.
_REVEAL_RE = re.compile(r'^Type of "(?P<expression>.+)" is "(?P<signature>.*)"$', re.DOTALL)


def infer_types(repo_root: Path, function: FunctionRef) -> dict[str, str] | None:
    """The parameter and return types of ``function`` as ``{name: type}`` plus ``"return"``.

    A fully annotated function returns its annotation strings without running anything:
    every positional and keyword-only parameter, ``*args`` and ``**kwargs`` when present,
    and the return must carry an annotation. Otherwise a copy of the function's file,
    with a ``reveal_type`` probe appended and its relative path preserved, is analysed
    with ``pyright --outputjson`` in a temporary directory, and the revealed signature
    is parsed for every parameter and ``"return"``.

    Returns ``None`` when any type is ``Unknown``, when the signature cannot be parsed,
    when the file cannot be staged, or when ``pyright`` is not installed (with a warning).
    Never raises: every failure mode means "no types".
    """
    annotated = _annotated_types(function.source)
    if annotated is not None:
        return annotated
    if shutil.which("pyright") is None:
        logger.warning("pyright is not installed; cannot infer types of %s", function.qualname)
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="pcc-infer-types-") as tmp:
            tmp_root = Path(tmp)
            target = _copy_with_probe(repo_root, function, tmp_root)
            if target is None:
                return None
            signature = _revealed_signature(tmp_root, target, function.qualname)
    except (OSError, UnicodeDecodeError) as error:
        logger.warning("cannot stage %s for pyright: %s", function.qualname, error)
        return None
    if signature is None:
        return None
    return _parse_signature(signature, function.qualname)


def _annotated_types(source: str) -> dict[str, str] | None:
    """The annotation strings of a fully annotated function, else ``None``.

    ``None`` when the source holds no function, or when any parameter (including
    ``*args`` and ``**kwargs`` when present) or the return lacks an annotation.
    Lambdas never qualify: they cannot carry a return annotation.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    func: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node
            break
    if func is None:
        return None
    params = [*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs]
    if func.args.vararg is not None:
        params.append(func.args.vararg)
    if func.args.kwarg is not None:
        params.append(func.args.kwarg)
    types: dict[str, str] = {}
    for param in params:
        if param.annotation is None:
            return None
        types[param.arg] = ast.unparse(param.annotation)
    if func.returns is None:
        return None
    types["return"] = ast.unparse(func.returns)
    return types


def _copy_with_probe(repo_root: Path, function: FunctionRef, tmp_root: Path) -> Path | None:
    """Copy the function's file under ``tmp_root`` with a ``reveal_type`` probe appended.

    The relative path is preserved, so relative imports keep resolving the same way.
    Returns ``None`` (with a warning) when the path escapes the repository, the file
    cannot be read or parsed, or the module already binds ``reveal_type``, which would
    make the probe call the wrong function.
    """
    parts = Path(function.path).parts
    if Path(function.path).is_absolute() or ".." in parts:
        logger.warning(
            "cannot infer types of %s: path %r escapes the repository",
            function.qualname,
            function.path,
        )
        return None
    try:
        text = (repo_root / function.path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        logger.warning(
            "cannot infer types of %s: cannot read %s (%s)",
            function.qualname,
            function.path,
            error,
        )
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        logger.warning(
            "cannot infer types of %s: cannot parse %s (%s)",
            function.qualname,
            function.path,
            error,
        )
        return None
    if _binds_reveal_type(tree):
        logger.warning(
            "cannot infer types of %s: %s already binds reveal_type",
            function.qualname,
            function.path,
        )
        return None
    target = tmp_root / function.path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + f"\nreveal_type({function.qualname})\n", encoding="utf-8")
    return target


def _binds_reveal_type(tree: ast.Module) -> bool:
    """Whether the module binds the name ``reveal_type`` itself."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "reveal_type":
                return True
        elif isinstance(node, ast.Import):
            if any(
                alias.asname == "reveal_type" or (alias.asname is None and alias.name.split(".")[0] == "reveal_type")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name != "*" and (alias.asname or alias.name) == "reveal_type" for alias in node.names):
                return True
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "reveal_type":
            return True
    return False


def _revealed_signature(tmp_root: Path, target: Path, qualname: str) -> str | None:
    """Run ``pyright --outputjson`` on ``target`` and return the probed signature.

    Returns ``None`` (with a warning) when pyright cannot start, times out, reports no
    JSON, or reveals no type for the probe. The exit code is ignored: pyright reports a
    non-zero exit for diagnostics, but the ``reveal_type`` answer is still there.
    """
    try:
        proc = subprocess.run(
            ["pyright", "--outputjson", str(target)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PYRIGHT_TIMEOUT_SECONDS,
            cwd=str(tmp_root),
            check=False,
        )
    except OSError as error:
        logger.warning("pyright could not be started for %s (%s)", qualname, error)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pyright timed out for %s", qualname)
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        logger.warning("pyright returned no JSON for %s (exit %s)", qualname, proc.returncode)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("generalDiagnostics"), list):
        logger.warning("pyright returned no diagnostics for %s", qualname)
        return None
    try:
        want = target.resolve()
    except OSError:
        logger.warning("cannot resolve the staged copy of %s", qualname)
        return None
    probe_line = target.read_text(encoding="utf-8").count("\n")
    for diagnostic in data["generalDiagnostics"]:
        if not isinstance(diagnostic, dict):
            continue
        try:
            same_file = Path(str(diagnostic.get("file", ""))).resolve() == want
        except OSError:
            continue
        if not same_file:
            continue
        match = _REVEAL_RE.match(str(diagnostic.get("message", "")))
        if match is None:
            continue
        if match.group("expression") != qualname and not _is_probe_line(diagnostic, probe_line):
            continue
        return match.group("signature")
    logger.warning("pyright revealed no type for %s", qualname)
    return None


def _is_probe_line(diagnostic: dict[object, object], probe_line: int) -> bool:
    """Whether the diagnostic sits on the appended probe line (1-based)."""
    try:
        region = diagnostic["range"]
        assert isinstance(region, dict)
        start = region["start"]
        assert isinstance(start, dict)
        line = start["line"]
        assert isinstance(line, int)
    except (KeyError, AssertionError):
        return False
    # Ranges are 0-based; the probe is the last line, which ends with a newline,
    # so its 1-based number equals the newline count of the staged file.
    return line == probe_line - 1


def _parse_signature(signature: str, qualname: str) -> dict[str, str] | None:
    """Split a revealed ``(params) -> return`` signature into parameter and return types.

    Returns ``None`` when any type contains ``Unknown``. A malformed signature also
    gives ``None`` (with a warning), never an exception.
    """
    text = signature.strip()
    params_end = _matching_paren(text)
    if params_end is None:
        logger.warning("cannot parse revealed type of %s: %r", qualname, signature)
        return None
    rest = text[params_end + 1 :].strip()
    if not rest.startswith("->"):
        logger.warning("cannot parse revealed type of %s: %r", qualname, signature)
        return None
    return_text = rest[2:].strip()
    if not return_text:
        logger.warning("cannot parse revealed type of %s: %r", qualname, signature)
        return None
    types: dict[str, str] = {}
    for chunk in _split_top_level(text[1:params_end]):
        if not chunk.strip():
            continue
        if _has_bare_lambda(chunk):
            # A default like ``= lambda a, b: a`` hides top-level commas from the
            # splitter, so the chunks no longer match the parameters.
            logger.warning("cannot parse revealed type of %s: %r", qualname, signature)
            return None
        param = _split_param(chunk)
        if param is None:
            if chunk.strip() in ("/", "*"):
                continue  # A positional-only or keyword-only marker, not a parameter.
            logger.warning("cannot parse revealed type of %s: %r", qualname, signature)
            return None
        name, type_text = param
        if not name.isidentifier() or not type_text:
            logger.warning("cannot parse revealed type of %s: %r", qualname, signature)
            return None
        if "Unknown" in type_text:
            return None
        types[name] = type_text
    if "Unknown" in return_text:
        return None
    types["return"] = return_text
    return types


def _matching_paren(text: str) -> int | None:
    """The index of the ``)`` closing the leading ``(``, quote-aware, else ``None``."""
    if not text.startswith("("):
        return None
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _split_top_level(text: str) -> list[str]:
    """Split ``text`` on commas outside brackets and string literals."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            current.append(char)
            if char == "\\":
                if index + 1 < len(text):
                    current.append(text[index + 1])
                    index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    parts.append("".join(current))
    return parts


def _split_param(chunk: str) -> tuple[str, str] | None:
    """The ``(name, type)`` of one ``name: type = default`` chunk.

    Returns ``None`` for a chunk without a top-level colon; the caller keeps ``/``
    and ``*`` markers and rejects anything else. The default is cut at the first
    top-level ``=``; ``*args`` keeps its plain name.
    """
    depth = 0
    quote: str | None = None
    colon: int | None = None
    index = 0
    while index < len(chunk):
        char = chunk[index]
        if quote is not None:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ":" and depth == 0:
            colon = index
            break
        index += 1
    if colon is None:
        return None
    name = chunk[:colon].strip().lstrip("*").strip()
    after = chunk[colon + 1 :]
    equals: int | None = None
    depth = 0
    quote = None
    index = 0
    while index < len(after):
        char = after[index]
        if quote is not None:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0 and after[index + 1 : index + 2] != "=":
            # A bare ``=`` starts the default; ``==`` cannot.
            equals = index
            break
        index += 1
    type_text = (after[:equals] if equals is not None else after).strip()
    return (name, type_text)


def _has_bare_lambda(chunk: str) -> bool:
    """Whether ``chunk`` holds a ``lambda`` outside brackets and string literals."""
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(chunk):
        char = chunk[index]
        if quote is not None:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif (char.isalpha() or char == "_") and depth == 0:
            start = index
            while index < len(chunk) and (chunk[index].isalnum() or chunk[index] == "_"):
                index += 1
            if chunk[start:index] == "lambda":
                return True
            continue
        index += 1
    return False


__all__ = ["infer_types"]
