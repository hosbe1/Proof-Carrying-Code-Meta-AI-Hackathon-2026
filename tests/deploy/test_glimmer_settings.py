"""Settings parsing for the Glimmer deployment (issue #63).

These tests are hermetic: no GPU, no network, and neither Modal nor vLLM is
installed or imported. Only ``deploy/glimmer_settings.py`` is imported.
The AST checks on ``deploy/modal_glimmer.py`` below are regression guards
for the image chain and the ``serve()`` decorator only; live container
behavior must be confirmed by the run issue, not by these tests.
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
    # The call must sit on the image build chain after pip_install and the
    # HF_HOME env step, so the container resolves the
    # `from deploy.glimmer_settings import ...` helpers with cached weights.
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    assert isinstance(receiver, ast.Call)
    assert isinstance(receiver.func, ast.Attribute) and receiver.func.attr == "env"
    pip_call = receiver.func.value
    assert isinstance(pip_call, ast.Call)
    assert isinstance(pip_call.func, ast.Attribute) and pip_call.func.attr == "pip_install"
    pip_args = {arg.value for arg in pip_call.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
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


def _static_int(node: ast.expr) -> int:
    """Evaluate a small static integer expression (literals, `*` products)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _static_int(node.left) * _static_int(node.right)
    raise AssertionError(f"expected a static integer expression, got {ast.dump(node)}")


def test_modal_image_caches_hf_weights_on_volume() -> None:
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    call = _image_add_local_call(tree)
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    assert isinstance(receiver, ast.Call)
    assert isinstance(receiver.func, ast.Attribute) and receiver.func.attr == "env"
    # The env step must point HF_HOME at the mounted volume, so a cold start
    # reuses the downloaded weights instead of fetching them again.
    assert len(receiver.args) == 1 and isinstance(receiver.args[0], ast.Dict)
    keys = receiver.args[0].keys
    values = receiver.args[0].values
    hf_index = next(index for index, key in enumerate(keys) if isinstance(key, ast.Constant) and key.value == "HF_HOME")
    hf_value = values[hf_index]
    assert ("VOLUME_MOUNT_PATH" in ast.dump(hf_value) and "/hf" in ast.dump(hf_value)) or (
        isinstance(hf_value, ast.Constant) and hf_value.value == "/models/pcc-glimmer/hf"
    )


def test_modal_function_allows_slow_weight_load() -> None:
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    decorator = _serve_function_decorator(tree)
    timeout = [keyword for keyword in decorator.keywords if keyword.arg == "startup_timeout"]
    assert len(timeout) == 1
    # Loading the weights can take tens of minutes; Modal must not kill the
    # runner (and with it the vLLM engine) after its default 300 seconds.
    assert _static_int(timeout[0].value) == 45 * 60


def _serve_web_server_decorator(tree: ast.Module) -> ast.Call:
    serve_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "serve"]
    assert len(serve_defs) == 1, "deploy/modal_glimmer.py must define serve()"
    web_calls = [
        node
        for node in serve_defs[0].decorator_list
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "web_server"
    ]
    assert len(web_calls) == 1, "serve() must carry exactly one @modal.web_server decorator"
    return web_calls[0]


def test_modal_web_server_waits_for_slow_weight_load() -> None:
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    decorator = _serve_web_server_decorator(tree)
    timeout = [keyword for keyword in decorator.keywords if keyword.arg == "startup_timeout"]
    assert len(timeout) == 1
    # The port health-check must wait as long as the runner: if loading the
    # weights takes more than the web_server timeout, the port check fails
    # even though the runner (and the vLLM engine) would have survived.
    assert _static_int(timeout[0].value) == 45 * 60


def test_deploy_docs_prime_the_hf_weight_cache() -> None:
    docs = (REPO_ROOT / "docs" / "deploy.md").read_text(encoding="utf-8")
    put_lines = [line.strip() for line in docs.splitlines() if "modal volume put" in line]
    assert put_lines, "docs/deploy.md must show how to prime the weight cache on the volume"
    # The image sets HF_HOME to /models/pcc-glimmer/hf, so only an upload
    # under /hf lands in the cache a cold start reuses.
    assert any(line.rstrip().endswith("/hf") for line in put_lines), (
        "docs/deploy.md must upload the base weights under /hf, the HF_HOME cache dir"
    )
    assert not any(line.rstrip().endswith("/weights") for line in put_lines), (
        "docs/deploy.md must not direct the base weights to /weights, outside the HF_HOME cache"
    )


def _serve_function(tree: ast.Module) -> ast.FunctionDef:
    serve_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "serve"]
    assert len(serve_defs) == 1, "deploy/modal_glimmer.py must define serve()"
    return serve_defs[0]


def test_serve_starts_vllm_in_background() -> None:
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    serve = _serve_function(tree)
    # With @modal.web_server the function must start the server in the
    # background and return; returning the command list never starts vLLM.
    assert serve.returns is None or (isinstance(serve.returns, ast.Constant) and serve.returns.value is None)
    popens = [
        node
        for node in ast.walk(serve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(popens) >= 1, "serve() must start the vLLM server with subprocess.Popen"
    assert all(
        not isinstance(node.value, (ast.List, ast.Tuple)) for node in ast.walk(serve) if isinstance(node, ast.Return)
    ), "serve() must not return the command list"
    assert "import subprocess" in source


def test_serve_surfaces_immediate_vllm_exit() -> None:
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    serve = _serve_function(tree)
    # serve() must keep the Popen handle so it can check for an early exit.
    popen_targets = [
        node.targets[0].id
        for node in ast.walk(serve)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "Popen"
    ]
    assert len(popen_targets) == 1, "serve() must assign the Popen handle to one name"
    handle = popen_targets[0]
    polls = [
        node
        for node in ast.walk(serve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "poll"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == handle
    ]
    assert polls, "serve() must poll the vLLM process for an immediate exit"
    assert any(
        isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name) and node.exc.func.id == "RuntimeError"
        for node in ast.walk(serve)
        if isinstance(node, ast.Raise)
    ), "serve() must raise RuntimeError with the exit code on immediate exit"
    assert "time.sleep" in source
