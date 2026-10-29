"""Step 4, fallback: translation by a model for whatever the transpiler does not support yet."""

from __future__ import annotations

from pcc.translate_ai.translate import translate

__all__ = ["translate"]
