"""Core data types, errors and serialization. Every other module depends on this package; it depends on nothing.

The types are defined in ``pcc.core.types``, which lists its public names in ``__all__``. This package re-exports all of
them, so adding a type touches only ``types.py``.
"""

from pcc.core import types as _types
from pcc.core.errors import NotCheckable, PccError, ToolError, UnsupportedConstruct
from pcc.core.serialize import dumps, from_data, loads, to_data
from pcc.core.types import *  # noqa: F403

__all__ = [
    *_types.__all__,
    "NotCheckable",
    "PccError",
    "ToolError",
    "UnsupportedConstruct",
    "dumps",
    "from_data",
    "loads",
    "to_data",
]
