"""LoRA SFT 1 retrain of Glimmer on Modal.

This is the script of the SFT 1 retrain: the same data and recipe as the
Fireworks attempt, rerun on Modal after the adapter download stayed
unavailable. The resulting adapter is the one served and evaluated.

Run from the repository root::

    modal run deploy/modal_sft.py --input runs/round0/rescored.jsonl

The input must be the round 0 rescore (301 rows); the run refuses any other
input. The adapter lands on the shared volume next to the serving deployment
(see ``docs/deploy.md``) and is served as ``glimmer-sft`` with
``PCC_GLIMMER_ADAPTERS=glimmer-sft=/adapters/sft1``.

Heavy dependencies (torch, transformers, peft) are imported inside
:func:`train`, which executes on Modal. Never import this module in tests:
Modal is an optional dependency that is not installed in CI.

Seed and execution note: ``SEED`` is a pin for deterministic reruns; the
original retrain log recorded no seed, so the pin applies to reruns only.
This script has not been executed end to end in CI; hermetic tests pin its
constants only, and run figures stay controller-reported in
``docs/results/training.md``.
"""

from __future__ import annotations

import modal  # type: ignore[import-not-found]  # optional, not a dev dependency

from deploy.glimmer_settings import VOLUME_MOUNT_PATH, VOLUME_NAME

APP_NAME = "pcc-glimmer-sft"
BASE_MODEL_ID = "meta-models/Muse-Glimmer-30B"
DEFAULT_INPUT = "runs/round0/rescored.jsonl"
INPUT_SHA256 = "c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1"
EXPECTED_ROWS = 301
SEED = 0
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
MAX_SEQ_LENGTH = 16384
IGNORE_INDEX = -100
TRAIN_DTYPE = "bf16"
GPU = "H200"
GRADIENT_CHECKPOINTING = True
ADAPTER_DIR = "adapters/sft1"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
TIMEOUT_SECONDS = 3600

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "peft", "accelerate")


@app.function(image=image, gpu=GPU, volumes={VOLUME_MOUNT_PATH: volume}, timeout=TIMEOUT_SECONDS)
def train(input_path: str = DEFAULT_INPUT) -> str:
    """Run the SFT 1 LoRA retrain and save the adapter to the shared volume.

    Uses seed 0, wraps the rows in a Dataset, and saves the model and the
    tokenizer into the adapter directory. The seed pin covers reruns; the
    original retrain log recorded no seed.
    """
    import hashlib
    import json

    digest = hashlib.sha256()
    rows: list[dict[str, object]] = []
    with open(input_path, "rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows.append(json.loads(line))
    if digest.hexdigest() != INPUT_SHA256 or len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Refusing to train: {input_path} has SHA-256 {digest.hexdigest()} "
            f"and {len(rows)} rows, expected {INPUT_SHA256} and {EXPECTED_ROWS}."
        )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(SEED)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, use_cache=not GRADIENT_CHECKPOINTING
    )
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    encoder = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if encoder.pad_token is None:
        encoder.pad_token = encoder.eos_token
    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=list(LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    def encode(row: dict[str, object]) -> dict[str, list[int]] | None:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        # The loss covers the assistant turn only: prompt ids are masked with IGNORE_INDEX.
        prompt_messages = messages[:-1]
        prompt_ids = encoder.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
        full_ids: list[int] = encoder.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if len(full_ids) > MAX_SEQ_LENGTH:
            # Rows longer than the length cap are dropped; none were dropped for SFT 1.
            return None
        labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return {"input_ids": full_ids, "labels": labels}

    encoded = [item for row in rows if (item := encode(row)) is not None]
    if len(encoded) != len(rows):
        raise ValueError(f"Dropped {len(rows) - len(encoded)} of {len(rows)} rows above {MAX_SEQ_LENGTH}.")

    class _RowDataset(torch.utils.data.Dataset):
        """Thin wrapper so the trainer receives a Dataset, not a plain list."""

        def __init__(self, items: list[dict[str, list[int]]]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self._items[index]

    train_dataset = _RowDataset(encoded)

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        pad = encoder.pad_token_id or 0
        padded_ids = [feature["input_ids"] + [pad] * (width - len(feature["input_ids"])) for feature in features]
        padded_labels = [feature["labels"] + [IGNORE_INDEX] * (width - len(feature["labels"])) for feature in features]
        batch = {"input_ids": torch.tensor(padded_ids), "labels": torch.tensor(padded_labels)}
        batch["attention_mask"] = (batch["input_ids"] != pad).long()
        return batch

    args = TrainingArguments(
        output_dir="/tmp/sft1-out",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        adam_beta1=ADAM_BETAS[0],
        adam_beta2=ADAM_BETAS[1],
        weight_decay=WEIGHT_DECAY,
        bf16=TRAIN_DTYPE == "bf16",
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        logging_steps=1,
        save_strategy="no",
        seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=collate)
    trainer.train()

    adapter_path = f"{VOLUME_MOUNT_PATH}/{ADAPTER_DIR}"
    model.save_pretrained(adapter_path)
    encoder.save_pretrained(adapter_path)
    volume.commit()
    return f"saved {ADAPTER_FILES[0]} and {ADAPTER_FILES[1]} to {VOLUME_NAME}:{ADAPTER_DIR}"
