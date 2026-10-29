"""The anti-cheat scan over Dafny proof attempts. Filled in by a later ticket."""

from __future__ import annotations


def anti_cheat(dafny_code: str, frozen_spec: str, *, assumptions_module: str = "") -> tuple[str, ...]:
    """Return every violation in ``dafny_code``; an empty tuple means it passes.

    Outside ``assumptions_module`` every one of these forms is a violation (an empty ``assumptions_module`` means no
    module may hold contracts):

    - ``assume``, with or without attributes (``assume {:axiom} false;`` included);
    - ``{:axiom}`` and ``@Axiom`` in any position;
    - body-less lemmas, functions and methods (Dafny warns on them, and they are contracts in disguise);
    - ``{:verify false}`` (Dafny 4.11 has no ``@`` spelling of it: ``@VerifyFalse`` does not resolve);
    - ``{:only}`` and ``@VerifyOnly``, which disable the verification of every other member of the file;
    - ``{:extern}``, a body-less method that Dafny takes on trust;
    - any change to ``frozen_spec``: the frozen spec must still appear unchanged in ``dafny_code``.

    Inside ``assumptions_module`` the only allowed form is a contract: a body-less lemma or function marked ``{:axiom}``
    or ``@Axiom``. Plain ``assume`` statements are reported everywhere, the module included.

    The forms and what Dafny 4.11 does with each are fixed in ``tests/verifier/cheats``.
    """
    raise NotImplementedError


__all__ = ["anti_cheat"]
