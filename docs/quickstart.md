# Quickstart

This path is CPU-first. It runs the deterministic mock backend, writes sealed
run artifacts, regenerates the local report, and builds a small evidence bundle.
It does not require a GPU, vLLM, internet access, or an external model server.

## Prerequisites

- Python 3.10 or newer.
- `uv` installed.
- A checkout of this repository.

Install the development dependencies:

```bash
uv sync --extra dev
```

Optional causal-transformer backends use extra dependencies and are covered
later in this guide.

## Run The CPU Smoke Benchmark

Use the checked-in smoke config:

```bash
uv run python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/small.yaml
```

The command prints a run directory like:

```text
artifacts/runs/small-1770000000000
```

Each benchmark run writes the required reproducibility files under that run
directory:

| File | Purpose |
| --- | --- |
| `requests.jsonl` | Per-request prompt, routing, response, quality, cache, and latency records. |
| `summary.json` | Aggregate quality, cache, latency, SLO, goodput, and workload metrics. |
| `config_resolved.yaml` | Fully resolved benchmark config. |
| `manifest.json` | Run metadata, git metadata, and artifact inventory. |

Raw run artifacts are intentionally ignored by git under `artifacts/runs/`.

## Generate A Report

Regenerate the report from whatever runs are present locally:

```bash
uv run python -m adapter_cache_bench.analysis.report \
  --runs-dir artifacts/runs
```

This writes:

- `reports/adapter-cache-tradeoffs.md`
- `reports/tables/*.csv`
- `reports/figures/*.png`

Reports are generated artifacts. Use them for local inspection and evidence
bundles; do not treat a local report as a public claim unless the underlying
runs, configs, and claim boundary are documented.

## Build An Evidence Bundle

Build a manifest for the smoke run:

```bash
uv run python -m adapter_cache_bench.analysis.evidence_bundle \
  --bundle-name quickstart-smoke \
  --run-glob "small-*" \
  --report reports/adapter-cache-tradeoffs.md \
  --table reports/tables/claim_evidence.csv
```

The bundle manifest lives at:

```text
evidence/quickstart-smoke/bundle_manifest.json
```

Evidence bundles record hashes, git metadata, included paths, missing required
files, and excluded raw artifacts. They do not copy `requests.jsonl` or backend
metric scrapes; those remain in `artifacts/runs/<run-id>/`.

See [evidence_bundles.md](evidence_bundles.md) for selectors, report/figure
inputs, and bundle policy.

## Run A Source-Backed CPU Eval

For a slightly more realistic CPU path, validate and run the bundled
source-backed JSONL fixture with the mock backend:

```bash
uv run python -m adapter_cache_bench.workloads.validate_dataset \
  --config configs/benchmark/source_eval.yaml
uv run python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/source_eval.yaml
```

This still uses `backend.kind: mock`, so it is useful for cache, routing, SLO,
and artifact sanity. It does not produce real causal-transformer quality.

For your own JSONL eval, copy `configs/benchmark/source_eval.yaml` to a new
config under `configs/benchmark/`, update `workload.dataset_path`, and validate
the config before running it. Keep reproducible benchmark settings in
`configs/benchmark/` instead of relying on one-off command-line overrides. The
record schema is documented in [eval_datasets.md](eval_datasets.md).

## Optional Local Causal-Transformer Path

If you want real local causal-transformer output without a model server, use
the Hugging Face causal-transformer backend:

```bash
uv sync --extra dev --extra real
uv run --extra real python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/source_eval_transformers.yaml
```

The default config uses a small CPU-capable causal-transformer model and only a
few requests. It may still need model files to be present locally or downloaded
by the Hugging Face stack. This path is optional and should not be required for
unit tests or CPU-only development.

## Optional vLLM Or External Server Path

Use vLLM or another OpenAI-compatible local server only when you need real
serving evidence. Start the server separately, then run a checked-in config
against it:

```bash
uv run python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/vllm_example.yaml
```

The source-backed vLLM config uses the same harness with more requests:

```bash
uv run python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/source_eval_vllm.yaml
```

For vLLM LoRA serving, start vLLM with `--enable-lora` and served model names
such as `qa-lora`, `json-lora`, `summary-lora`, and `code-lora`, then use the
LoRA overlay:

```bash
uv run python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/source_eval_vllm.yaml \
           configs/benchmark/source_eval_vllm_lora_qwen.yaml
```

These paths are optional. They may require GPU, model downloads, adapter files,
server launch parameters, and server-side metrics. Real-serving claims should
cite the model, adapters, hardware, request count, run count, metric scope, and
the evidence bundle. See [vllm.md](vllm.md),
[model_backends.md](model_backends.md), and [gcloud_vllm.md](gcloud_vllm.md)
for the serving runbooks.

Optional server integration tests stay opt-in:

```bash
RUN_VLLM_TESTS=1 uv run pytest tests/test_optional_integrations.py -q
```

Do not set `RUN_VLLM_TESTS=1` unless a compatible server is available.

## Next CPU Suites

After the smoke path works, the main reproducible CPU suites are:

```bash
uv run python -m adapter_cache_bench.bench.run_matrix \
  --config configs/benchmark/benchmark_v0_mock.yaml
uv run python -m adapter_cache_bench.bench.run_matrix \
  --config configs/benchmark/regime_v0_mock.yaml \
  --sweep-name regime-v0-mock \
  --resume \
  --continue-on-error \
  --max-runs 600 \
  --max-requests 75000 \
  --estimated-seconds-per-run 1 \
  --max-estimated-gpu-hours 1
```

`benchmark_v0_mock.yaml` is the frozen CPU reproducibility suite.
`regime_v0_mock.yaml` is the simulator-backed regime-science suite. Both are
CPU/mock evidence; neither proves production vLLM behavior or GPU capacity.

## Useful Checks

Run the unit and formatting checks before publishing code changes:

```bash
uv run pytest tests -q
uv run ruff check .
uv run ruff format . --check
```

These checks must not require GPU, vLLM, internet access, or an external model
server. GPU and external-server tests remain explicit opt-ins.
