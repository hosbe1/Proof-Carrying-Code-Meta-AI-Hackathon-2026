# Deploy: serve Glimmer on Modal

Glimmer serves the evaluation before and after training, and the demo. The
deployment runs on Modal with vLLM's OpenAI-compatible server on one GPU
(`H100` by default), scales to zero after 15 idle minutes, and runs only
while a job needs it. Nothing is deployed by this repository's CI; deploying
is a manual step owned by the run tickets (#76, #77, #79, #57).

## Contract (for #57 and #70)

| Name | Value |
|---|---|
| Hugging Face base model id (provisional, pending #3) | `meta-models/Muse-Glimmer-30B` |
| Modal secret holding the bearer token | `pcc-glimmer-api` (key `PCC_GLIMMER_API_KEY`) |
| Modal volume caching the base weights | `pcc-glimmer`, mounted at `/models/pcc-glimmer` |
| Served model name | `glimmer` |
| Endpoint | authenticated web endpoint, bearer token from the secret |
| Scale-down | 15 idle minutes (`scaledown_window` 900 s) |
| Scale | at most one container (`max_containers=1`), so a cold start never fans out beyond one GPU |
| LoRA | always `--enable-lora --max-lora-rank 16` |

The base model id is a placeholder until #3 confirms it. It lives in one
place, `DEFAULT_BASE_MODEL` in `deploy/glimmer_settings.py`, so correcting it
is a one-line change.

## Settings

`deploy/glimmer_settings.py` parses the environment and imports neither Modal
nor vLLM. All settings with their defaults:

| Variable | Default | Meaning |
|---|---|---|
| `PCC_GLIMMER_BASE_MODEL` | `meta-models/Muse-Glimmer-30B` (provisional) | Base weights to serve |
| `PCC_GLIMMER_GPU` | `H100` | GPU class for the Modal function |
| `PCC_GLIMMER_MAX_MODEL_LEN` | `32768` | vLLM `--max-model-len`; sampling rounds 0 and 1 may override with `8192` |
| `PCC_GLIMMER_QUANTIZATION` | empty (disabled) | vLLM `--quantization`; only `fp8` is supported |
| `PCC_GLIMMER_ADAPTERS` | empty (base model only) | `name=source` entries, see below |
| `PCC_GLIMMER_MODEL` | `glimmer` | Model name clients request; set to `glimmer-sft` for the trained model |

`PCC_GLIMMER_MAX_MODEL_LEN` must be a positive integer, `PCC_GLIMMER_QUANTIZATION`
must be empty or `fp8`; anything else fails fast with a clear error.

Adapters: each `name=source` pair is served as one `--lora-modules name=path`
entry, in the given order. A source starting with `/` is a path under the
mounted volume (resolved to `/models/pcc-glimmer/<source>`); any other source
is a Hugging Face model id. Volume paths are checked at server startup: a
missing path stops the deployment with an error that names the path. Examples:

```bash
export PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1,other=some-org/other-lora"
```

## Deploy

1. Create the bearer-token secret (once; rotate by recreating it):

   ```bash
   export PCC_GLIMMER_API_KEY="$(openssl rand -hex 32)"
   modal secret create pcc-glimmer-api PCC_GLIMMER_API_KEY="$PCC_GLIMMER_API_KEY"
   ```

2. Prime the weight cache on the volume (once; subsequent deploys reuse it):

   ```bash
   modal volume put pcc-glimmer /local/path/to/weights /hf
   ```

   The image sets `HF_HOME` to `/models/pcc-glimmer/hf`, so only files under
   the volume-relative `/hf` (container path `/models/pcc-glimmer/hf`) land
   in the cache a cold start reuses. If you have no local copy,
   skip this step: the first cold start downloads the weights itself (about
   60 GB) and later starts reuse them. If the volume was primed under the
   previous volume-relative `/weights`, that copy sits outside `HF_HOME`
   and is ignored: a cold start downloads the weights again. Re-prime
   those files under `/hf` instead and drop the stale `/weights` copy.

   Upload trained adapters the same way, e.g. under `/adapters/sft1`, when
   their source is a volume path rather than a Hugging Face id.

3. Export the settings for the run and deploy from the repository root:

   ```bash
   export PCC_GLIMMER_GPU=H100
   export PCC_GLIMMER_MAX_MODEL_LEN=32768
   export PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1"
   modal deploy deploy/modal_glimmer.py
   ```

    Deploy reads the environment when it runs; changing a setting means
    re-deploying.

    The image bakes the deploy-time `PCC_GLIMMER_*` settings (never the
    `PCC_GLIMMER_API_KEY`, which arrives via the secret at runtime) into the
    container after the `HF_HOME` step and before `add_local_python_source`,
    because the settings module reloads inside the container where those
    variables are unset. Without this, `PCC_GLIMMER_QUANTIZATION` and
    `PCC_GLIMMER_ADAPTERS` would be silently ignored and vLLM would always
    run with defaults.

    The image points `HF_HOME` at the mounted volume
    (`/models/pcc-glimmer/hf`), so a cold start reuses weights already
    downloaded into the volume-relative `/hf` instead of downloading them again. The cache only
    helps if the volume already holds a prior download in the volume-relative
    `/hf`; on an
    empty volume the first cold start still downloads them. The serving function allows 45 minutes
    to start (`startup_timeout`), since loading the weights takes tens of
     minutes; `serve()` starts the vLLM server in the background and returns,
     as `@modal.web_server` requires. The image also sets
     `VLLM_USE_FLASHINFER_SAMPLER=0`, so vLLM skips the FlashInfer sampler
     JIT build that needs `nvcc`, which the slim image does not ship.

## One test request

After deploy, Modal prints the web endpoint URL. Test it with the served
model name (`glimmer`, or the adapter name for a trained model):

```bash
curl "$PCC_GLIMMER_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $PCC_GLIMMER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "glimmer", "messages": [{"role": "user", "content": "Say ok."}]}'
```

Expect a `200` response with a completion. For the trained model, use
`"model": "glimmer-sft"` and set `PCC_GLIMMER_MODEL=glimmer-sft` for the run.

## Warm-up before a run

A scaled-to-zero deployment needs a few minutes to load the weights. Before a
run, send exactly one warm-up request (the test request above, or one short
completion against the model the run uses) and wait for the `200`. Do not
retry in parallel: queued requests during a cold start scale out to more
containers. Only then start the sampling or evaluation job, so cold-start
latency is not billed as run time.

`max_containers=1` caps containers, not client concurrency. `serve()` carries
`@modal.concurrent(max_inputs=64)`, so the single container serves up to 64
concurrent client requests at once. Measure the
concurrency-64 throughput with 64 concurrent client requests against the
single container (vLLM batches them server-side); do not raise the container
cap to reach that concurrency. Record the deploy log (no `ModuleNotFoundError`),
the base URL without the key, the warm-up `200`, the BF16 and FP8 tokens per
second, and the Modal spend in the run issue.

## Measured throughput (run #76)

Base model `meta-models/Muse-Glimmer-30B` (provisional id, pending #3),
vLLM 0.32.0, one H100, served name `glimmer`. Cold start with cached weights:
255–346 s. Throughput measured at concurrency 64 with prompts of about 2,029
tokens and 512 output tokens each (`ignore_eos`), one fresh server per
precision. These numbers are valid only for that id: if #3 confirms a
different id, repeat the concurrency-64 measurement on the confirmed id
before using the numbers for cost estimates:

| Precision | Wall time | Output tokens/s |
|---|---|---|
| BF16 | 30.6 s | 1,072 |
| FP8 (`--quantization fp8`) | 22.1 s | 1,485 |

At about $3.95 per H100 hour that is roughly $1.00 (BF16) or $0.74 (FP8) per
million output tokens. Modal spend for the whole run day, including the failed
starts while debugging, was $10.30 (covered by credits, $0 billed), which
exceeds the 1 H100 hour (about $4) cap of this run: the overrun comes from
debugging before `max_containers=1` (queued warm-up requests scaled to nine
H100 containers) and is recorded here as an overrun, not as compliance.
Closing this run with that overrun needs explicit PO acceptance in the run
issue. The measurement slice alone stayed within the cap: two fresh servers
at most 346 s cold start each plus 30.6 s and 22.1 s of measurement wall time
is under 13 minutes (about 0.2 H100 hours). Later runs stay within the cap
with `max_containers=1`, a single warm-up request, and no parallel retries
during a cold start.

## Stop

The app scales to zero by itself after 15 idle minutes, so stopping is usually
unneeded. To stop it immediately or remove it:

```bash
modal app stop pcc-glimmer    # stop running containers, keep the app
modal app delete pcc-glimmer  # remove the app; the volume and secret stay
```

Deleting the app does not delete the `pcc-glimmer` volume or the
`pcc-glimmer-api` secret; delete those explicitly if the cached weights or the
token must go.

Every run records its stop in its run issue: the `modal app stop` output,
or `modal app list` showing no running containers after the scale-down
window. For run #76, the issue records the controller stopping the pre-fix
deploy, but no separate stop output or app-list snapshot was captured for the
final measurement servers, which were left to the 15-minute scale-down; this
gap is recorded here, and closing the run needs PO acceptance of it.

## Cost

GPUs bill per second while containers run, and $0 while scaled to zero. An
`H100` is about $3.95 per GPU hour (check the current
[Modal pricing page](https://modal.com/pricing) before a run, rates change).
Rule of thumb: ~10 H100 hours fit in $40, ~15 hours in $60; see the caps in
`docs/plan.md` (round 1 sampling $40, evaluation and demo serving $60). Every
run records its Modal spend in its issue.

## Fallback: SGLang

If vLLM does not load Glimmer, serve it with SGLang instead, which also
exposes an OpenAI-compatible API (`--served-model-name glimmer`,
`--enable-lora`, `--max-lora-rank 16`, `--context-length` for the max model
length, `--quantization fp8` when enabled, LoRA adapters via
`--lora-target-modules` / the adapter registry). The settings module,
environment variables, secret, volume and mount path stay the same; only the
server command in `deploy/modal_glimmer.py` changes. Clients keep using
`PCC_GLIMMER_BASE_URL` / `PCC_GLIMMER_API_KEY` unchanged.
