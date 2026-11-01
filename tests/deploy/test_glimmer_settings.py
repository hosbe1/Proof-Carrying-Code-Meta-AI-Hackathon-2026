"""Settings parsing for the Glimmer deployment (issue #63).

These tests are hermetic: no GPU, no network, and neither Modal nor vLLM is
installed or imported. Only ``deploy/glimmer_settings.py`` is imported.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from deploy.glimmer_settings import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GPU,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL,
    DEFAULT_QUANTIZATION,
    MAX_LORA_RANK,
    SCALEDOWN_WINDOW_SECONDS,
    SECRET_NAME,
    SERVED_MODEL_NAME,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
    Adapter,
    GlimmerSettings,
    check_volume_paths,
    load_settings,
    parse_adapters,
    vllm_server_args,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "deploy" / "glimmer_settings.py"
MODAL_APP_PATH = REPO_ROOT / "deploy" / "modal_glimmer.py"


def test_defaults_need_no_environment() -> None:
    settings = load_settings({})
    assert settings == GlimmerSettings()
    assert settings.base_model == DEFAULT_BASE_MODEL == "meta-models/Muse-Glimmer-30B"
    assert settings.gpu == DEFAULT_GPU == "H100"
    assert settings.max_model_len == DEFAULT_MAX_MODEL_LEN == 32768
    assert settings.quantization == DEFAULT_QUANTIZATION == ""
    assert settings.adapters == ()
    assert settings.model == DEFAULT_MODEL == "glimmer"


def test_default_gpu_is_h100_and_default_max_model_len_is_32768() -> None:
    assert DEFAULT_GPU == "H100"
    assert DEFAULT_MAX_MODEL_LEN == 32768
    settings = load_settings({})
    assert settings.gpu == "H100"
    assert settings.max_model_len == 32768


def test_settings_module_imports_neither_modal_nor_vllm() -> None:
    source = SETTINGS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "modal" not in imported
    assert "vllm" not in imported
    # Dynamic imports would defeat the hermetic guarantee as well.
    assert "__import__" not in source
    assert "importlib" not in imported


def test_missing_adapter_volume_path_error_names_the_path(tmp_path: Path) -> None:
    missing = "/missing-adapter-path-63/adapter"
    adapters = (Adapter(name="glimmer-sft", source=missing),)
    with pytest.raises(FileNotFoundError) as excinfo:
        check_volume_paths(adapters, mount=str(tmp_path))
    assert missing in str(excinfo.value)


def test_existing_adapter_volume_path_passes_check(tmp_path: Path) -> None:
    present = tmp_path / "adapters" / "sft1"
    present.mkdir(parents=True)
    adapters = (Adapter(name="glimmer-sft", source="/adapters/sft1"),)
    check_volume_paths(adapters, mount=str(tmp_path))


def test_two_adapters_give_two_lora_modules_in_order() -> None:
    adapters = parse_adapters("glimmer-sft=adapters/sft1,other=adapters/sft2")
    assert [(adapter.name, adapter.source) for adapter in adapters] == [
        ("glimmer-sft", "adapters/sft1"),
        ("other", "adapters/sft2"),
    ]
    settings = load_settings({"PCC_GLIMMER_ADAPTERS": "glimmer-sft=adapters/sft1,other=adapters/sft2"})
    args = vllm_server_args(settings)
    lora_index = args.index("--lora-modules")
    assert args[lora_index + 1 : lora_index + 3] == ["glimmer-sft=adapters/sft1", "other=adapters/sft2"]


def test_unset_adapters_serve_base_model_without_lora_modules() -> None:
    settings = load_settings({})
    assert settings.adapters == ()
    args = vllm_server_args(settings)
    assert "--lora-modules" not in args


def test_blank_adapters_behaves_like_unset() -> None:
    assert parse_adapters("") == ()
    assert parse_adapters("   ") == ()
    assert load_settings({"PCC_GLIMMER_ADAPTERS": ""}).adapters == ()


def test_hugging_face_id_adapter_needs_no_path_check() -> None:
    adapters = parse_adapters("glimmer-sft=some-org/glimmer-sft-lora")
    assert len(adapters) == 1
    assert not adapters[0].is_volume_path
    # A Hugging Face id is accepted without any path check, even offline.
    check_volume_paths(adapters, mount="/nonexistent-mount")


def test_absolute_source_is_a_volume_path() -> None:
    adapter = Adapter(name="glimmer-sft", source="/adapters/sft1")
    assert adapter.is_volume_path
    assert adapter.target("/models/pcc-glimmer") == "/models/pcc-glimmer/adapters/sft1"
    assert adapter.lora_entry("/models/pcc-glimmer") == "glimmer-sft=/models/pcc-glimmer/adapters/sft1"


def test_relative_source_is_a_hugging_face_id() -> None:
    adapter = Adapter(name="other", source="adapters/sft2")
    assert not adapter.is_volume_path
    assert adapter.target("/models/pcc-glimmer") == "adapters/sft2"
    assert adapter.lora_entry("/models/pcc-glimmer") == "other=adapters/sft2"


def test_malformed_adapter_entries_name_the_entry() -> None:
    cases = (
        ("no-equals-sign", "no-equals-sign"),
        ("=source-only", "=source-only"),
        ("name-only=", "name-only="),
        ("ok=a,=bad", "=bad"),
    )
    for raw, entry in cases:
        with pytest.raises(ValueError, match=re.escape(entry)):
            parse_adapters(raw)


def test_max_model_len_must_be_positive() -> None:
    for raw in ("0", "-1", "not-a-number"):
        with pytest.raises(ValueError, match="PCC_GLIMMER_MAX_MODEL_LEN"):
            load_settings({"PCC_GLIMMER_MAX_MODEL_LEN": raw})


def test_max_model_len_override_for_sampling_rounds() -> None:
    settings = load_settings({"PCC_GLIMMER_MAX_MODEL_LEN": "8192"})
    assert settings.max_model_len == 8192
    assert "--max-model-len" in vllm_server_args(settings)


def test_quantization_supports_fp8_or_empty() -> None:
    assert load_settings({}).quantization == ""
    assert load_settings({"PCC_GLIMMER_QUANTIZATION": "fp8"}).quantization == "fp8"
    with pytest.raises(ValueError, match="PCC_GLIMMER_QUANTIZATION"):
        load_settings({"PCC_GLIMMER_QUANTIZATION": "int8"})


def test_quantization_flag_only_present_when_enabled() -> None:
    assert "--quantization" not in vllm_server_args(load_settings({}))
    assert "fp8" in vllm_server_args(load_settings({"PCC_GLIMMER_QUANTIZATION": "fp8"}))


def test_environment_overrides() -> None:
    settings = load_settings(
        {
            "PCC_GLIMMER_BASE_MODEL": "some-org/some-model",
            "PCC_GLIMMER_GPU": "H100",
            "PCC_GLIMMER_MODEL": "glimmer-sft",
        }
    )
    assert settings.base_model == "some-org/some-model"
    assert settings.gpu == "H100"
    assert settings.model == "glimmer-sft"


def test_modal_contract_constants_for_downstream_runs() -> None:
    assert SECRET_NAME == "pcc-glimmer-api"
    assert VOLUME_NAME == "pcc-glimmer"
    assert VOLUME_MOUNT_PATH == "/models/pcc-glimmer"
    assert SERVED_MODEL_NAME == "glimmer"
    assert MAX_LORA_RANK == 16
    assert SCALEDOWN_WINDOW_SECONDS == 15 * 60


def test_vllm_args_carry_the_fixed_serving_parameters() -> None:
    args = vllm_server_args(load_settings({}))
    assert "--served-model-name" in args
    assert args[args.index("--served-model-name") + 1] == "glimmer"
    assert "--enable-lora" in args
    assert args[args.index("--max-lora-rank") + 1] == "16"
    assert args[args.index("--max-model-len") + 1] == "32768"


def _image_add_local_call(tree: ast.Module) -> ast.Call:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_python_source"
    ]
    assert len(calls) == 1, "deploy/modal_glimmer.py must call add_local_python_source exactly once"
    call = calls[0]
    assert len(call.args) == 1 and isinstance(call.args[0], ast.Constant) and call.args[0].value == "deploy"
    return call


def test_modal_app_bundles_deploy_sources_for_container() -> None:
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    call = _image_add_local_call(tree)
    # The call must sit on the image build chain after pip_install, so the
    # container resolves the `from deploy.glimmer_settings import ...` helpers.
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    assert isinstance(receiver, ast.Call)
    assert isinstance(receiver.func, ast.Attribute) and receiver.func.attr == "pip_install"
    pip_args = {arg.value for arg in receiver.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    assert {"vllm", "huggingface_hub"} <= pip_args
    assert "from deploy.glimmer_settings import" in source
    image_assigns = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "image" for target in node.targets)
    ]
    assert len(image_assigns) == 1
    assert call in ast.walk(image_assigns[0].value)


def _serve_function_decorator(tree: ast.Module) -> ast.Call:
    serve_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "serve"]
    assert len(serve_defs) == 1, "deploy/modal_glimmer.py must define serve()"
    decorators = serve_defs[0].decorator_list
    app_calls = [
        node
        for node in decorators
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "function"
    ]
    assert len(app_calls) == 1, "serve() must carry exactly one @app.function decorator"
    return app_calls[0]


def test_modal_app_caps_containers_at_one_gpu() -> None:
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    decorator = _serve_function_decorator(tree)
    cap = [keyword for keyword in decorator.keywords if keyword.arg == "max_containers"]
    assert len(cap) == 1
    assert isinstance(cap[0].value, ast.Constant) and cap[0].value.value == 1
    # The cap must apply to the serving function with its image, so a stray
    # keyword elsewhere cannot satisfy this test.
    image_kw = [keyword for keyword in decorator.keywords if keyword.arg == "image"]
    assert len(image_kw) == 1
    assert isinstance(image_kw[0].value, ast.Name) and image_kw[0].value.id == "image"


def test_modal_app_defines_module_level_app_without_being_imported() -> None:
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "App"
        for node in tree.body
    ), "deploy/modal_glimmer.py must define module-level `app = modal.App(...)`"
