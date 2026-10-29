"""JSON round-trip for frozen dataclasses such as the core types in ``pcc.core.types`` and ``pcc.config.Config``.

Supported field types: str, int, float, bool, None, StrEnum, nested frozen dataclasses, ``tuple[X, ...]``, fixed
``tuple[A, B]``, ``Mapping[str, V]`` and ``X | None``. That is all the core and config types use; anything else raises
``TypeError`` so it is caught in review.

Reading is strict: a bool is never an int or a float, a float is built only from a JSON number, a tuple only from a JSON
list, a mapping only from a JSON object with string keys, unknown keys and unknown enum values are rejected, and NaN or
infinity are rejected both ways. Missing fields take their defaults. Wrong types, unknown names and unknown values raise
``TypeError``; NaN, infinity and malformed JSON text raise ``ValueError``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import types
import typing
from collections.abc import Mapping
from enum import Enum
from typing import Any, NoReturn


def to_data(value: Any) -> Any:
    """Convert a core value into plain JSON data."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_data(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [to_data(v) for v in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("only mappings with string keys are serializable")
        return {key: to_data(item) for key, item in value.items()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"not serializable: {type(value).__name__}")


def from_data[T](cls: type[T], data: Any) -> T:
    """Rebuild a core dataclass of type ``cls`` from plain JSON data."""
    return typing.cast(T, _build(cls, data))


def dumps(value: Any) -> str:
    """Serialize as JSON. Raises ``ValueError`` for NaN and infinity, which JSON cannot represent."""
    return json.dumps(to_data(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def loads[T](cls: type[T], text: str) -> T:
    return from_data(cls, json.loads(text, parse_constant=_reject_constant))


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not valid JSON")


def _build(tp: Any, data: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if data is None:
            return None
        if len(args) != 1:
            raise TypeError(f"unsupported union: {tp}")
        return _build(args[0], data)
    if origin is tuple:
        tuple_args = typing.get_args(tp)
        if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
            (item, _) = tuple_args
            if not isinstance(data, list):
                raise TypeError(f"expected a list for {tp}, got {type(data).__name__}")
            return tuple(_build(item, entry) for entry in data)
        if not isinstance(data, list) or len(data) != len(tuple_args):
            raise TypeError(f"expected a list of length {len(tuple_args)} for {tp}, got {data!r}")
        return tuple(_build(item_tp, entry) for item_tp, entry in zip(tuple_args, data, strict=True))
    if origin is dict or origin is Mapping:
        (key_tp, value_tp) = typing.get_args(tp)
        if not isinstance(data, dict):
            raise TypeError(f"expected an object for {tp}, got {type(data).__name__}")
        return {_build(key_tp, key): _build(value_tp, item) for key, item in data.items()}
    if isinstance(tp, type) and issubclass(tp, Enum):
        try:
            return tp(data)
        except ValueError as error:
            raise TypeError(f"unknown value for {tp.__name__}: {data!r}") from error
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _build_dataclass(tp, data)
    if tp is bool or tp is str or tp is int:
        # bool is a subclass of int, so an int must be checked against bool explicitly.
        if not isinstance(data, tp) or (tp is int and isinstance(data, bool)):
            raise TypeError(f"expected {tp.__name__}, got {type(data).__name__}")
        return data
    if tp is float:
        if isinstance(data, bool) or not isinstance(data, int | float):
            raise TypeError(f"expected float, got {type(data).__name__}")
        return float(data)
    raise TypeError(f"unsupported type: {tp}")


def _build_dataclass(cls: Any, data: Any) -> Any:
    if not isinstance(data, dict):
        raise TypeError(f"expected an object for {cls.__name__}, got {type(data).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(str(key) for key in data if key not in names)
    if unknown:
        raise TypeError(f"unknown fields for {cls.__name__}: {', '.join(unknown)}")
    hints = _type_hints(cls)
    # Missing fields keep their defaults; a required field that is missing fails in the constructor.
    kwargs = {name: _build(hints[name], data[name]) for name in names if name in data}
    return cls(**kwargs)


@functools.cache
def _type_hints(cls: Any) -> dict[str, Any]:
    # Resolved once per class. Names resolve in the module that defines cls, so a field can use types from other
    # modules.
    return typing.get_type_hints(cls)
