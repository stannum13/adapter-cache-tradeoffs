# Research plan

This repo now has a reproducible mock benchmark, real streamed vLLM runs,
committed result figures, strict external-eval preflight, model-family sweep
support, server reset hooks, and an adapter/cache metrics postprocessor. The
remaining work is mostly about collecting stronger evidence, not adding another
mock-only feature.

For the line between current evidence and benchmark-quality claims, see
[benchmark_quality_plan.md](benchmark_quality_plan.md). The short version:
`benchmark_v0` is frozen as a reproducible suite, while broad model-quality
claims still require independent external data and repeated isolated vLLM runs.

## 1. External evaluation data

Goal: replace public-domain-style engineering fixtures with a larger held-out
dataset whose provenance and license are suitable for public research claims.

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

Status: command path is implemented. The default config uses the generated
500-row public-domain-style fixture to keep the run reproducible; replace
`workload.dataset_path` with independently curated records before making public
dataset-quality claims.

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
make vllm-exhaustive-overlap
make vllm-exhaustive-adapter-count
```

Status: reset/warmup hooks are implemented through `BackendConfig`, and
manifests record `server_reset.log` when enabled. Use this path for the next
isolated vLLM rerun.

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

Status: sweep expansion supports `matrix.models` and per-family adapter maps.
The evidence is not complete until a second family has compatible specialist
and multitask adapters trained with the same SFT protocol.

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

The checker reports the five lanes above as `ok`, `needs_evidence`, or
`missing`. A `needs_evidence` result is acceptable for the multi-model lane
until a second model family has actually been trained and served; it should not
be described as completed evidence in the paper text before then.
