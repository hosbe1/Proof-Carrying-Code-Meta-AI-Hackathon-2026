"""Modal SFT 1 retrain script pins (issue #78).

The retrain ran on Modal with the same data and recipe as the Fireworks
attempt, and its adapter is the one served and evaluated. These tests are
hermetic: no GPU, no network, and Modal is neither installed nor imported.
Only ``deploy/glimmer_settings.py`` is imported; ``deploy/modal_sft.py`` is
read as source and its module-level recipe constants are checked.
"""

from __future__ import annotations

import ast
from pathlib import Path

from deploy.glimmer_settings import DEFAULT_BASE_MODEL, VOLUME_MOUNT_PATH, VOLUME_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
MODAL_SFT_PATH = REPO_ROOT / "deploy" / "modal_sft.py"

INPUT_SHA256 = "c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1"


def _source() -> str:
    return MODAL_SFT_PATH.read_text(encoding="utf-8")


def _constants() -> dict[str, object]:
    tree = ast.parse(_source())
    constants: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
    return constants


def test_script_exists_and_parses() -> None:
    ast.parse(_source())


def test_base_model_matches_serving_base_model() -> None:
    assert _constants()["BASE_MODEL_ID"] == DEFAULT_BASE_MODEL == "meta-models/Muse-Glimmer-30B"


def test_training_data_is_pinned_to_round0_rescored() -> None:
    constants = _constants()
    assert constants["DEFAULT_INPUT"] == "runs/round0/rescored.jsonl"
    assert constants["INPUT_SHA256"] == INPUT_SHA256
    assert constants["EXPECTED_ROWS"] == 301


def test_lora_recipe_matches_run_record() -> None:
    constants = _constants()
    assert constants["LORA_RANK"] == 16
    assert constants["LORA_ALPHA"] == 32
    assert constants["LORA_TARGET_MODULES"] == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_optimizer_recipe_matches_run_record() -> None:
    constants = _constants()
    assert constants["EPOCHS"] == 2
    assert constants["BATCH_SIZE"] == 8
    assert constants["LEARNING_RATE"] == 1e-4
    assert constants["ADAM_BETAS"] == (0.9, 0.95)
    assert constants["WEIGHT_DECAY"] == 0.0


def test_precision_hardware_and_length_pins() -> None:
    constants = _constants()
    assert constants["TRAIN_DTYPE"] == "bf16"
    assert constants["GPU"] == "H200"
    assert constants["GRADIENT_CHECKPOINTING"] is True
    assert constants["MAX_SEQ_LENGTH"] == 16384
    assert constants["IGNORE_INDEX"] == -100


def test_adapter_lands_on_the_shared_volume() -> None:
    constants = _constants()
    assert constants["ADAPTER_DIR"] == "adapters/sft1"
    assert constants["ADAPTER_FILES"] == ("adapter_config.json", "adapter_model.safetensors")
    assert VOLUME_NAME == "pcc-glimmer"
    assert VOLUME_MOUNT_PATH == "/models/pcc-glimmer"
    source = _source()
    assert "adapters/sft1" in source


def test_volume_constants_come_from_glimmer_settings() -> None:
    tree = ast.parse(_source())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "deploy.glimmer_settings":
            imported.update(alias.name for alias in node.names)
    assert "VOLUME_NAME" in imported
    assert "VOLUME_MOUNT_PATH" in imported


def test_modal_app_defines_module_level_app_without_being_imported() -> None:
    tree = ast.parse(_source())
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "App"
        for node in tree.body
    ), "deploy/modal_sft.py must define module-level `app = modal.App(...)`"


def test_train_function_runs_on_the_modal_app() -> None:
    tree = ast.parse(_source())
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == "train"
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
            for decorator in node.decorator_list
        )
        for node in tree.body
    ), "deploy/modal_sft.py must define `@app.function def train(...)`"


def test_loss_is_masked_to_the_assistant_turn() -> None:
    source = _source().lower()
    assert "assistant" in source
    assert "ignore_index" in source


def test_long_rows_are_dropped() -> None:
    assert "drop" in _source().lower()


def _train_node() -> ast.FunctionDef:
    tree = ast.parse(_source())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            return node
    raise AssertionError("deploy/modal_sft.py must define `def train(...)`")


def test_seed_is_pinned() -> None:
    assert _constants()["SEED"] == 0


def test_train_sets_seed_for_reproducibility() -> None:
    train = _train_node()
    calls = [node for node in ast.walk(train) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Name) and call.func.id == "set_seed" for call in calls), (
        "train must call set_seed for deterministic shuffling"
    )
    assert any(any(keyword.arg == "seed" for keyword in call.keywords) for call in calls), (
        "train must pass seed into TrainingArguments"
    )


def test_train_uses_a_dataset_object() -> None:
    segment = ast.get_source_segment(_source(), _train_node()) or ""
    assert "Dataset" in segment, "train must wrap encoded rows in a Dataset object"
    assert "train_dataset" in segment


def test_train_saves_tokenizer_with_adapter() -> None:
    segment = ast.get_source_segment(_source(), _train_node()) or ""
    assert segment.count("save_pretrained") >= 2, (
        "train must save the model and the tokenizer into the adapter directory"
    )


def test_script_holds_no_credentials() -> None:
    tree = ast.parse(_source())
    suspicious = {"api_key", "apikey", "token", "password", "passwd", "secret"}
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.lower() in suspicious:
                raise AssertionError(f"deploy/modal_sft.py assigns a credential-like name: {target.id}")
    lowered = _source().lower()
    for secret_word in ("api_key", "api-key", "password", "secret"):
        assert secret_word not in lowered
