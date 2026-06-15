# Benchmark quality plan

This repository is currently a benchmark harness with early real-serving
evidence. It should not be described as a finished benchmark until the suite,
data, scoring, and run protocol below are frozen and repeated.

## Current status

| area | current status | benchmark-quality requirement |
| --- | --- | --- |
| Workload generation | Controlled synthetic and source-backed fixtures exist. | Freeze a named suite with fixed records, seeds, prompt layouts, tenant fields, and overlap distributions. |
| Quality scoring | Task-specific heuristic scoring is implemented. | Use deterministic, documented scorers with task-level failure examples and source provenance. |
| Real serving | vLLM runs exist for L4 and H100 paths. | Restart or isolate the server per condition, record startup logs, and repeat each condition. |
| Cache evidence | Simulator metrics and vLLM server-level counters exist. | Report simulator cache accounting and server counters side by side, with metric scope clearly labeled. |
| Capacity evidence | One L4 fails 8/10 LoRAs at 4096 context; one H100 80GB serves 10 LoRAs. | Record a GPU/model/context/adapter-count capacity table with exact failure messages and startup logs. |
| External validity | Public-domain-style and source-backed fixtures exist. | Add an independently curated public eval fixture before making broad model-quality claims. |

## Benchmark v0 definition

`benchmark_v0` is the first suite that should be treated as frozen. It has two
tiers.

### Tier 1: CPU reproducibility suite

Purpose: validate the systems hypotheses without requiring a GPU or internet.

Command:

```bash
make benchmark-v0-mock
```

Config:

- [configs/benchmark/benchmark_v0_mock.yaml](../configs/benchmark/benchmark_v0_mock.yaml)

Frozen dimensions:

- workloads: `shared_doc_qa`, `mixed_tasks_same_doc`,
  `prompt_layout_ablation`, `low_overlap_control`;
- routers: `semantic`, `multitask`, `sticky_session`, `cache_aware`, `oracle`;
- cache models: `standard_lora`, `activated_lora`, `copy_on_write`;
- seeds: `17`, `23`, `31`;
- block size: `8`;
- request count: `96`;
- tenants: `4`;
- sessions: `12`.

Expected interpretation:

- `low_overlap_control` should reduce or remove the cache-locality advantage.
- `prompt_layout_ablation` should expose the late-specialization effect:
  document-before-invocation is the layout where activated-style caching can
  preserve shared-prefix locality.
- `standard_lora` should show more fragmentation than activated/copy-on-write
  cache simulations under shared-prefix multi-adapter workloads.

### Tier 2: real vLLM evidence suite

Purpose: validate that the cache and capacity tradeoffs survive real model
serving.

Minimal commands, assuming a local vLLM server with the documented model names:

```bash
make vllm-source-eval-expanded-qwen7b
make vllm-source-eval-expanded-lora-trained-qwen7b
make vllm-source-eval-expanded-lora-multitask-qwen7b
```

Capacity commands are infrastructure-specific and should be recorded in
[docs/large_model_results.md](large_model_results.md), not hidden behind
undocumented shell history.

Required real-serving metadata:

- GPU type and memory;
- VM/machine type or local host shape;
- vLLM version;
- model id;
- context length;
- loaded adapter names and paths;
- LoRA rank and adapter count;
- `gpu_memory_utilization`;
- prefix caching flag;
- startup success/failure;
- vLLM startup logs for KV cache size and max concurrency;
- `/metrics` counter deltas.

## Canonical metrics

Every table that claims benchmark evidence should include:

- request count;
- mean, p50, p95, and p99 TTFT;
- mean and p95 E2E latency where available;
- SLO attainment;
- request throughput;
- token throughput where available;
- mean quality;
- quality-adjusted goodput;
- benchmark-side cache hit rate;
- server prefix-cache hit rate when using vLLM;
- memory-token footprint or vLLM KV cache capacity;
- adapter distribution.

## External eval requirements

The next credibility jump is an independent public eval fixture. It must have:

- at least 500 records;
- balanced `qa`, `json`, `summary`, and `code` tasks;
- repeated shared documents with stable `shared_prefix_id`;
- both prompt layouts;
- source/provenance fields;
- tenant and trust-group fields;
- deterministic scoring rules;
- license notes suitable for public release.

The repo already has a preflight command:

```bash
make validate-external-eval
```

The current default external-eval template is still an engineering fixture. It
is useful for exercising the pipeline, not for final paper-quality claims.

## Claim discipline

Acceptable current claim:

> Adapter specialization has a measurable cache and memory footprint. In this
> harness and early vLLM evidence, specialist LoRAs improved source-backed task
> quality, but adapter count and cache namespace fragmentation changed serving
> headroom and prefix-cache reuse.

Claims to avoid until stronger evidence exists:

- specialists are universally better than multitask adapters;
- activated LoRA is validated in a real vLLM kernel path;
- the source-backed fixture is a standard external benchmark;
- H100 latency results are directly comparable to L4 latency results without
  controlling server configuration and loaded adapter count.

## Remaining work

1. Freeze `benchmark_v0` outputs into one canonical CSV.
2. Add confidence intervals over repeated serving runs.
3. Add an independently curated external eval fixture.
4. Run real vLLM conditions with server reset per condition.
5. Add a paper-style capacity frontier table generated from structured data.
6. Keep failed runs in the documentation when they explain the frontier.
