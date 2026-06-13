# The hidden cache footprint of specialization

Specialist adapters improve quality, but every routing decision is also a cache decision. This repo benchmarks when semantic routing, sticky routing, cache-aware routing, multitask adapters, and activated adapters win under shared-prefix workloads.

## Research question

When is model/adaptor specialization worth its KV-cache footprint?

The thesis: specialist adapters can improve task quality, but they can fragment prefix-cache reuse because cache namespaces often depend on adapter identity. A serving system should jointly optimize quality, cache locality, latency SLOs, and tenant isolation.

## Why naive semantic routing can hurt

Semantic routing sends QA to a QA adapter, JSON extraction to a JSON adapter, and code/documentation tasks to a code adapter. Under a shared long document, that looks right locally, but standard LoRA-style prefix cache keys include adapter identity. The same document prefix can be cached once per adapter instead of once per trust group, increasing prefill work, TTFT, and memory footprint.

## Quickstart

```bash
uv sync --extra dev
uv run pytest tests -q
uv run python -m specialization_cache_frontier.bench.run_workload --config configs/benchmark/small.yaml
```

Outputs:

- `artifacts/runs/{run_id}/requests.jsonl`
- `artifacts/runs/{run_id}/summary.json`
- `artifacts/runs/{run_id}/config_resolved.yaml`
- `reports/specialization-cache-frontier.md`
- `reports/figures/*.png`

## Expected plots

- quality vs p95 TTFT, bubble size = goodput, color = router policy
- prefix/cache hit rate by router policy and cache model
- quality-adjusted goodput by router policy
- memory token footprint by cache model
- prompt layout ablation
- adapter strategy frontier

## Repo structure

- `configs/`: benchmark, router, cache, and workload YAMLs
- `src/specialization_cache_frontier/cache/`: whitespace tokenizer and block prefix cache simulators
- `src/specialization_cache_frontier/routing/`: random, semantic, sticky, cache-aware, and oracle policies
- `src/specialization_cache_frontier/backends/`: mock backend plus optional vLLM client stub
- `src/specialization_cache_frontier/bench/`: workload and matrix runners
- `src/specialization_cache_frontier/analysis/`: plots, report, and Pareto helpers
- `src/specialization_cache_frontier/tiny_causal_transformer/`: minimal decoder-only causal transformer fundamentals
- `src/specialization_cache_frontier/physical_ai_analogue/`: scene-cache simulator and mapping notes

## Run the mock benchmark

```bash
uv run python -m specialization_cache_frontier.bench.run_workload --config configs/benchmark/small.yaml
uv run python -m specialization_cache_frontier.bench.run_matrix --config configs/benchmark/full.yaml
```

The default path requires no GPU and no internet after dependencies are installed.

## Plug in vLLM

`VLLMBackend` is an OpenAI-compatible client stub. Configure `backend.base_url`, `backend.api_key`, `backend.model`, and adapter names, then extend response parsing for your server metrics. Integration tests should be skipped unless `RUN_VLLM_TESTS=1`.

## Limitations

The first pass uses whitespace tokenization, approximate block caching, synthetic quality priors, and a deterministic mock backend. It is designed to make cache-routing tradeoffs reproducible before validating them on real serving stacks.

## Physical AI analogue

The same issue appears in VLA serving: repeated visual/proprioceptive scene tokens map to a world-state cache, skill adapters map to specialization, and goodput maps to success-rate-adjusted control Hz. See `src/specialization_cache_frontier/physical_ai_analogue/README.md`.

