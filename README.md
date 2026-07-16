# Adapter Cache Tradeoffs

Adapter Cache Tradeoffs is a benchmark harness for testing when specialist LoRA
adapters are worth their cache, memory, and tail-latency cost in causal
transformer serving runtimes.

The repository keeps the experiment code separate from serving runtimes. The
current harness can run deterministic CPU/mock sweeps locally, talk to
OpenAI-compatible model servers, record per-request JSONL, and preregister the
next SGLang-based experiment without vendoring upstream source.

## Current Status

| Area | Status |
| --- | --- |
| Correctness pass | Live dispatch-state routing, quality-adjusted SLO goodput, tenant/trust-group isolation scoring, and simulated-versus-server cache metric naming have been fixed on the public refocus branch. |
| Upstream substrate | SGLang, Vidur, and vLLM are pinned in [UPSTREAM.md](UPSTREAM.md). No upstream source is vendored. |
| E001 | Preregistered in [experiments/e001/experiment.md](experiments/e001/experiment.md). Canonical SGLang evidence has not been run yet. |
| Local smoke | `./scripts/reproduce_e001.sh smoke` runs a CPU/mock E001 smoke sweep and writes ignored artifacts under `artifacts/runs/`. |
| Canonical run | `./scripts/reproduce_e001.sh canonical --dry-run` validates the planned serving sweep budget before any GPU/server run. |

## Question

When does adapter-aware routing improve quality-adjusted serving capacity once
prefix locality, live load, adapter residency, and tail latency are measured in
a real serving runtime?

## Method

The canonical experiment compares routing strategies under fixed workload and
serving conditions:

- runtime/base-model routing;
- semantic specialist routing;
- sticky-session routing;
- cache-aware routing without active-load state;
- single multitask adapter routing;
- live-state cache-aware specialist routing;
- oracle routing as an upper-bound diagnostic only.

The primary metric is `quality_adjusted_slo_goodput`, defined as:

```text
sum(quality_i * I[ttft_i <= SLO]) / wall_time
```

The treatment must route near dispatch time and use live queue or in-flight
state, not historical assignment counts.

## Reproduce

Set up the local dev environment:

```bash
uv sync --extra dev
```

Run the regular local acceptance checks:

```bash
uv run pytest tests -q
uv run ruff check .
uv run ruff format . --check
```

Run the E001 smoke path:

```bash
./scripts/reproduce_e001.sh smoke
```

Inspect the canonical plan without launching a model server:

```bash
./scripts/reproduce_e001.sh canonical --dry-run
```

Run the canonical serving sweep only after starting the pinned SGLang server and
the required adapters:

```bash
./scripts/reproduce_e001.sh canonical --resume --continue-on-error
```

## Implementation Map

| Path | Purpose |
| --- | --- |
| `src/adapter_cache_bench/routing/` | Router policies and live session/dispatch state. |
| `src/adapter_cache_bench/cache/` | Simulator-side prefix/cache models and tokenizer approximations. |
| `src/adapter_cache_bench/backends/` | Mock, local Transformers, and OpenAI-compatible serving clients. |
| `src/adapter_cache_bench/bench/` | Workload, concurrent, matrix, and exhaustive sweep runners. |
| `src/adapter_cache_bench/analysis/` | Reports, evidence bundles, claim tables, SLO, Pareto, and plots. |
| `configs/` | Benchmark, router, cache, and workload YAMLs. |
| `experiments/e001/` | Preregistered E001 question, configs, and reproduction entrypoint. |
| `docs/` | Evidence policy, historical result notes, model-backend docs, and limitations. |

## Evidence Boundary

E001 has not produced canonical SGLang results yet. The current public branch
supports local smoke verification, dry-run budget validation, and historical
evidence review. Do not read the historical vLLM tables below as the result of
E001.

Historical runs are scoped to available summaries, manifests, run docs, and
server notes because some predate the current `status.json` lifecycle contract.
See [docs/legacy_evidence_policy.md](docs/legacy_evidence_policy.md) and
[docs/claim_ladder.md](docs/claim_ladder.md).

### Historical vLLM Evidence

The strongest historical cache-locality result is a reset-isolated vLLM sweep
on Qwen2.5-7B with one L4. It is useful prior evidence, not a replacement for
the preregistered SGLang experiment.

| Historical result | Value |
| --- | ---: |
| Requests | 400 |
| Reset-isolated runs | 10 |
| Medium-overlap server prefix hit rate | 26.4% |
| High-overlap server prefix hit rate | 83.8% |
| p95 TTFT reduction from 50% to 95% overlap | 666.0 ms |
| SLO attainment lift | 10.0 pp |
| Quality-adjusted goodput lift | 62.3% |

The broader historical 1.5B streamed vLLM sweep compared specialist and
multitask strategies over 7,520 requests and 112 sweep runs. Full context is in
[docs/real_eval_results.md](docs/real_eval_results.md); 7B and model-family
notes are in [docs/large_model_results.md](docs/large_model_results.md).

## Limitations

See [docs/limitations.md](docs/limitations.md) for the current release
boundary. The most important constraints are:

- CPU/mock smoke runs are systems checks, not model-quality evidence.
- Simulator cache metrics are not server prefix-cache counters.
- Whitespace tokenization is only a simulator approximation.
- Historical vLLM results are prior evidence and may not satisfy the current
  evidence-bundle lifecycle contract.
- Canonical SGLang evidence, Vidur screening, and vLLM replication remain
  pending.

## License

MIT. See [LICENSE](LICENSE).
