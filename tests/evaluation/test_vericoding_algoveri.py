"""AlgoVeri tasks keep the Vericoding anti-cheat contract (issue #181).

The frozen spec and anti-cheat rules apply unchanged: the contract (requires and
ensures plus signatures) is frozen and a cheating answer never reaches Dafny.
"""

from __future__ import annotations

import json
from pathlib import Path

from pcc.core import Usage
from pcc.evaluation import datasets, vericoding
from pcc.evaluation.datasets import Task
from pcc.models.client import Completion, Message
from pcc.models.fake import Call

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ALGOVERI_BYTES = (FIXTURES / "algoveri-dafny.jsonl").read_bytes()


def _algoveri_tasks() -> tuple[Task, ...]:
    rows = [json.loads(line) for line in ALGOVERI_BYTES.decode("utf-8").splitlines() if line.strip()]
    return datasets._parse_algoveri(("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8"))


class _SingleAnswerClient:
    """A client answering every prompt with one scripted answer."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[Call] = []

    def complete(
        self,
        model: str,
        messages: tuple[Message, ...],
        *,
        step: str,
        schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        self.calls.append(Call(model, messages, step, schema, temperature, max_tokens))
        usage = Usage(step=step, model=model, input_tokens=0, output_tokens=0, cost_usd=0.2, seconds=1.0)
        return Completion(self._text, usage)


def test_cheating_algoveri_answer_never_reaches_dafny(tmp_path: Path) -> None:
    (task,) = [task for task in _algoveri_tasks() if task.id == "algoveri/abs_val"]
    frozen = vericoding.task_spec_of(task)
    calls: list[str] = []

    def spy(answer: str, prefix: str) -> tuple[bool, float, str]:
        calls.append(answer)
        return True, 0.1, ""

    cheated = frozen.replace("ensures r == AbsVal(x)", "ensures true", 1) + "  r := 0;\n}\n"
    assert vericoding._is_cheat(cheated, frozen)
    out = tmp_path / "runs" / "algoveri-cheat"
    vericoding.run_harness(
        split="algoveri",
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        tasks=(task,),
        client=_SingleAnswerClient(cheated),
        verify_fn=spy,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert calls == []
