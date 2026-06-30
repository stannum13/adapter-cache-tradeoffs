# vLLM and local model servers

The default benchmark path uses `MockBackend` and requires no GPU. The optional
`VLLMBackend` uses an OpenAI-compatible `/chat/completions` endpoint. For a
generic local server, set `backend.kind: openai_compatible`; for vLLM-specific
configs, use `backend.kind: vllm`.

## Start a Base-Model Server

Example shape:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

Run the source-backed benchmark against that server:

```bash
make vllm-source-eval-l4-qwen
```

## Start a Multi-LoRA Server

vLLM's OpenAI-compatible server selects a registered LoRA adapter through the
request `model` field. Start vLLM with `--enable-lora` and one or more
`--lora-modules` entries:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --lora-modules \
    qa-lora=uditjain/lori-qwen2.5-1.5b-medical \
    json-lora=uditjain/lori-qwen2.5-1.5b-medical \
    summary-lora=uditjain/lori-qwen2.5-1.5b-medical \
    code-lora=uditjain/lori-qwen2.5-1.5b-medical
```

Then run:

```bash
make vllm-source-eval-lora-qwen
```

`configs/benchmark/source_eval_vllm_lora_qwen.yaml` maps benchmark adapter IDs
to those served model names:

```yaml
backend:
  adapter_model_names:
    qa: qa-lora
    json: json-lora
    summary: summary-lora
    code: code-lora
```

The public LoRA listed above is a serving smoke adapter with a vLLM-compatible
PEFT config (`modules_to_save: null`), not a task-specialist adapter trained for
this benchmark. Replace those module paths with your own QA, JSON, summary, and
code adapters for task-quality claims.

See [model_backends.md](model_backends.md) for the backend matrix.

## Run an Optional Smoke Benchmark

```bash
RUN_VLLM_TESTS=1 uv run pytest tests/test_optional_integrations.py -q
make vllm-example
```

`vllm_example.yaml` is intentionally small. `source_eval_vllm.yaml` runs the
source-backed eval bundle against the same OpenAI-compatible endpoint:

```bash
make vllm-source-eval
```

## Run Concurrent Load

The sequential runner is useful for quality and cache-accounting sanity. Use
`run_concurrent` when you need wall-clock goodput and SLO behavior under bounded
request concurrency:

```bash
make vllm-heldout-xlarge-lora-trained-qwen15b-concurrent
```

Concurrent configs set:

```yaml
backend:
  max_concurrency: 8
  request_spacing_ms: 0
```

The runner still writes `requests.jsonl`, `summary.json`,
`config_resolved.yaml`, and `manifest.json`. Its throughput and goodput metrics
use wall-clock run duration rather than the sum of per-request latencies.

For TTFT-sensitive studies, enable OpenAI-compatible streaming:

```yaml
backend:
  stream: true
```

With streaming enabled, `ttft_ms` is measured at the first non-empty content
chunk and `e2e_ms` is measured when the stream completes. Without streaming,
`ttft_ms` is a conservative whole-response latency proxy because a plain
OpenAI-compatible response does not expose first-token timing. The full
concurrency frontier can be run with:

```bash
make vllm-overnight-frontier-streaming
```

The broader exhaustion suite covers prompt layout, controlled overlap, adapter
count, tenant isolation, and repeated-seed confidence checks:

```bash
make vllm-exhaustive-all
```

For cleaner per-condition server metrics, configure a reset hook. The runner
executes `backend.server_reset_command`, waits for `backend.server_warmup_url`,
then scrapes `/metrics` before sending benchmark requests:

```yaml
backend:
  server_reset_command: ./scripts/restart_local_vllm_lora.sh
  server_reset_timeout_s: 300
  server_warmup_url: http://localhost:8000/health
  server_warmup_timeout_s: 300
  server_warmup_interval_s: 2
```

The reset command is intentionally environment-specific. Use it when you need
isolated vLLM prefix-cache counters per run; leave it unset for faster frontier
sweeps.

The repo includes `configs/benchmark/local_vllm_reset.yaml` as a local overlay
for this path:

```bash
uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep \
  --config configs/benchmark/exhaustive_overlap_vllm_streaming.yaml \
           configs/benchmark/local_vllm_reset.yaml
```

## Run the Minimal G8 Bridge

Use `configs/benchmark/vllm_bridge_reset.yaml` for the first Branch A -> G8
real-server bridge. It is a reset-isolated matrix with three workloads
(`controlled_overlap`, `mixed_tasks_same_doc`, and `low_overlap_control`), two
serving strategies (`specialists` and `multitask`), and two cache conditions
(`warm` and `prefix_disabled`). The bridge is exploratory and single-seed by
default; add repeated seeds only after the dry-run budget fits the available GPU
window.

Dry-run and budget check:

```bash
uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep \
  --config configs/benchmark/vllm_bridge_reset.yaml \
  --sweep-name vllm-bridge-reset-g8 \
  --dry-run \
  --max-runs 12 \
  --max-requests 300 \
  --estimated-seconds-per-run 180 \
  --max-estimated-gpu-hours 1
```

For a local GPU/Docker run, serve five LoRA model names and let the bridge reset
the server before each child:

```bash
export LORA_MODULES="qa-lora=/adapters/qwen15b-qa json-lora=/adapters/qwen15b-json summary-lora=/adapters/qwen15b-summary code-lora=/adapters/qwen15b-code multitask-lora=/adapters/qwen15b-multitask"

uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep \
  --config configs/benchmark/vllm_bridge_reset.yaml \
  --sweep-name vllm-bridge-reset-g8 \
  --resume \
  --max-runs 12 \
  --max-requests 300 \
  --estimated-seconds-per-run 180 \
  --max-estimated-gpu-hours 1
```

For a GCP L4 run, create/setup/start the VM and tunnel `localhost:8000` as in
[gcloud_vllm.md](gcloud_vllm.md), export `ACB_CLOUD_*` provenance variables,
then layer the cloud reset overlay:

```bash
uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep \
  --config configs/benchmark/vllm_bridge_reset.yaml \
           configs/benchmark/gcloud_7b_lora_bridge_reset.yaml \
  --sweep-name vllm-bridge-reset-g8-gcp7b \
  --resume \
  --max-runs 12 \
  --max-requests 300 \
  --estimated-seconds-per-run 240 \
  --max-estimated-gpu-hours 1
```

Every completed child run should have `requests.jsonl`, `summary.json`,
`config_resolved.yaml`, `manifest.json`, `status.json`, `server_reset.log`, and
backend metric snapshots or metric error files. Manifests should record
`backend_model`, `base_url`, `adapter_model_names`, `stream`, `max_concurrency`,
git metadata, and a `cloud` block when `ACB_CLOUD_*` variables are set. Treat
cache-mechanism claims as client-observed only unless the `summary.json` backend
metric deltas include the vLLM prefix-cache counters needed for the claim.

Configure `backend.base_url`, `backend.model`, and adapter metadata for your
server. Use the mock backend for unit tests, CI, and CPU-only development. vLLM
responses are scored with the benchmark's task metrics (`qa`, `json`,
`summary`, and `code`) when `ground_truth` is present in the request record.

If no vLLM server is available, `configs/benchmark/source_eval_transformers.yaml`
runs the same benchmark harness through a local Hugging Face causal LM backend:

```bash
make transformers-source-eval
```

## Metrics

The backend records wall-clock request latency from the client. For production
serving studies, enable streaming TTFT and also scrape server-side Prometheus
metrics with `MetricsClient`; join them with `requests.jsonl` by run timestamp
or request metadata.

When `/metrics` snapshots are available, `summary.json` includes raw
`backend_metrics` deltas. The analysis loader also exposes:

- `server_prefix_cache_queries`
- `server_prefix_cache_hits`
- `server_prefix_cache_hit_rate`
- `server_prompt_tokens_cached`

Vanilla vLLM prefix-cache metrics are server-level counters. Adapter-aware
per-namespace cache counters still require serving-layer instrumentation or
per-condition server resets plus benchmark-side cache-model accounting.

Generate the side-by-side adapter/cache evidence table with:

```bash
make adapter-metrics
```

This writes `reports/tables/adapter_cache_metrics.csv`. The table aggregates
request-level benchmark cache accounting by adapter and joins server-level vLLM
prefix-cache deltas by run. The `server_cache_metric_scope` column is
`per_condition_reset` when the run manifest includes `server_reset.log`,
`server_process_window` when metrics were scraped without a reset, and
`unavailable` when no server metrics are present.
