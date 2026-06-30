# Research plan

This repo now has a reproducible mock benchmark, real streamed vLLM runs,
committed result figures, strict external-eval preflight, a completed
Qwen/TinyLlama model-family sweep, server reset hooks, and an adapter/cache
metrics postprocessor. The remaining work is mostly about collecting stronger
external and standard-benchmark evidence, not adding another mock-only feature.

For the line between current evidence and benchmark-quality claims, see
[benchmark_quality_plan.md](benchmark_quality_plan.md). The short version:
`benchmark_v0` is frozen as a reproducible suite, while broad model-quality
claims still require separately curated benchmark data and repeated isolated
vLLM runs.

## 1. External evaluation data

Goal: use a larger held-out dataset whose provenance and license are suitable
for public research claims.

Acceptance criteria:

- At least 500 records.
- Balanced `qa`, `json`, `summary`, and `code` task types.
- Repeated shared documents with explicit `shared_prefix_id`.
- Both prompt layouts represented.
- Tenant/trust-group fields intentionally assigned.
- `make validate-external-eval` passes after pointing the template config at the
  dataset.

Run path:

```bash
make validate-external-eval
make vllm-external-eval
```

Status: the 500-row source-backed fixture has been served through the
model-family vLLM path. The default config uses
`data/eval/external_public_domain_eval.jsonl`. Replace
`workload.dataset_path` with separately curated records before making
standard-benchmark claims.

## 2. Per-condition vLLM cache isolation

Goal: make server-side prefix-cache metrics cleaner by resetting vLLM between
conditions that are intended to be independent.

Acceptance criteria:

- Configure `backend.server_reset_command`.
- Configure `backend.server_warmup_url`.
- Confirm every run manifest includes `server_reset.log`.
- Confirm `summary.json` includes nonzero `server_prefix_cache_queries`,
  `server_prefix_cache_hits`, and `server_prefix_cache_hit_rate`.

Run path:

```bash
make vllm-exhaustive-overlap-reset
make vllm-exhaustive-adapter-count-reset
```

Status: reset/warmup hooks are implemented through `BackendConfig`, and
manifests record `server_reset.log` when enabled. Use these reset-overlay
targets for the next isolated vLLM rerun.

## 3. Multi-model comparison

Goal: check whether the adapter/cache tradeoff holds beyond Qwen2.5-1.5B.

Acceptance criteria:

- Train compatible specialist and multitask adapters for at least one additional
  causal transformer family.
- Serve each adapter as a distinct vLLM model name.
- Add the model and adapter map to
  `configs/benchmark/model_family_vllm_template.yaml`.
- Run the same held-out eval with the same SLOs.

Run path:

```bash
make vllm-model-family
```

Status: completed for `Qwen/Qwen2.5-1.5B-Instruct` and
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` on the 500-row source-backed fixture.
Each family ran specialists and multitask across three seeds, for 12 vLLM runs
and 6,000 total requests. `make model-family-summary` regenerates
`reports/tables/model_family_summary.csv` from the run manifests. Next
credibility step: repeat with a stronger non-Qwen family and a separately
curated public benchmark fixture.

## 4. Adapter-aware serving metrics

Goal: move from benchmark-side simulation of adapter cache namespaces to
server-side visibility.

Options:

- Instrument vLLM locally to export adapter/model labels on prefix-cache
  counters.
- Run hard-isolated one-adapter-at-a-time server conditions and compare against
  benchmark-side cache accounting.
- Add a metrics postprocessor that joins request metadata with server counter
  windows.

Acceptance criteria:

- Results include server-level and benchmark-model cache metrics side by side.
- Docs clearly state which metrics are isolated and which are trend evidence.

Status: `make adapter-metrics` writes
`reports/tables/adapter_cache_metrics.csv`, joining benchmark-side per-adapter
cache accounting with server-level vLLM prefix-cache deltas and a metric-scope
label.

## 5. Public result refresh

Goal: keep the GitHub snapshot current without committing raw artifacts.

Acceptance criteria:

- `make report` regenerates local report/figures.
- Selected publication-worthy figures are copied into `docs/figures/`.
- `docs/real_eval_results.md` summarizes the run count, request count, and
  headline table.
- CPU checks pass.

Verification:

```bash
uv run pytest tests -q
uv run ruff check .
uv run ruff format . --check
```

Status: `make report`, `make whitepaper-figure`, and `make adapter-metrics`
refresh publication-facing derived artifacts without committing raw run logs.

## Readiness check

Run this before pushing a public research snapshot:

```bash
make research-readiness
```

The checker reports implementation preflights and evidence lanes as `ok`,
`needs_evidence`, or `missing`. The multi-model lane is now evidence-backed by
observed `model-family-vllm` run manifests. The capacity-frontier lane checks
that the structured records include startup failures and a larger-GPU success.
