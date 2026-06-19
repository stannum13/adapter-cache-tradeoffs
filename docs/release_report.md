# When is specialization worth its cache footprint?

## Abstract

Specialist adapters can improve task quality, but every adapter routing decision
also changes the cache namespace seen by a causal transformer serving stack.
This benchmark studies that tradeoff under shared-prefix workloads.

## Core question

When is model or adaptor specialization worth its KV-cache footprint?

## Why this matters

Naive semantic routing sends each task to its best specialist. Under repeated
long prefixes, that can fragment prefix-cache reuse across adapters and increase
TTFT even when the semantic choice is locally correct.

## Experimental setup

The deterministic mock backend is a systems simulator: it isolates routing,
prefix-cache locality, finite KV budget, eviction, prompt layout, and SLO
behavior from GPU and serving noise. Real model-server runs use the same
JSONL workloads with `backend.kind: vllm` or another OpenAI-compatible
backend.

### Evidence classes

- `mock` / `mock-causal-transformer`: 182 runs, 17552 requests, mean quality 0.892, mean p95 TTFT 77.5 ms.
- `unknown` / `unknown`: 116 runs, 5221 requests, mean quality 0.837, mean p95 TTFT 111.4 ms.
- `vllm` / `Qwen/Qwen2.5-1.5B-Instruct`: 199 runs, 15036 requests, mean quality 0.448, mean p95 TTFT 1228.5 ms.
- `vllm` / `Qwen/Qwen2.5-7B-Instruct`: 46 runs, 3432 requests, mean quality 0.274, mean p95 TTFT 1033.7 ms.
- `vllm` / `TinyLlama/TinyLlama-1.1B-Chat-v1.0`: 6 runs, 3000 requests, mean quality 0.375, mean p95 TTFT 87.4 ms.

### Claim ladder

The maintained public claim boundary lives in
[docs/claim_ladder.md](claim_ladder.md). Current supported claims
must cite model/server, request count, run count, and metric scope.

- Reset-isolated 7B cache locality: 400 requests across 10 runs. Moving from 50% to 95% shared-prefix overlap raised server prefix-cache hit rate by 57.4 percentage points, reduced mean p95 TTFT by 666.0 ms (41.5%), lifted SLO attainment by 10.0 percentage points, and raised QAG by 62.3%.

## Workloads

The benchmark includes shared document QA, mixed tasks over the same document,
multi-turn agent sessions, a low-overlap negative control, and prompt layout
ablations.

## Router policies

Policies include random, semantic, sticky session, cache-aware, and oracle routing.

## Cache models

Models include standard LoRA-style adapter namespaces, optimistic base sharing,
activated-LoRA-style late specialization, and copy-on-write deltas.

## Results

Best quality-adjusted goodput in the current artifact set is `cache_aware` with `activated_lora` on `mixed_tasks_same_doc`. Mean quality is 0.915, p95 TTFT is 19.3 ms, and fragmentation index is 1.00. `document_before_instruction` mean TTFT is 105.4 ms. `instruction_before_document` mean TTFT is 165.2 ms.

### Decision rule

Treat specialization as worthwhile only when it improves quality-adjusted
goodput under the TTFT SLO after accounting for cache memory. In this
repo that means comparing both `quality_adjusted_goodput` and
`quality_adjusted_goodput_per_memory_token`, while checking that the
fragmentation index and SLO attainment do not regress beyond the serving
budget for the workload.

### Interpretation

- Best aggregate cache strategy: `base_shared` with mean quality-adjusted goodput 13.940.
- Best cache-footprint efficiency: `activated_lora` with mean quality-adjusted goodput per memory token 0.011626.
- Prompt layout matters: `document_before_instruction` is 58.2 ms lower mean TTFT than `instruction_before_document` in the current artifact set.
- Highest eviction pressure: `standard_lora` on `mixed_tasks_same_doc` with 12 evictions.
- Repeated-seed leader: `cache_aware` with `activated_lora` on `mixed_tasks_same_doc`.

Generated figures:

- `reports/figures/quality_vs_p95_ttft.png`
- `reports/figures/cache_hit_rate_by_policy_model.png`
- `reports/figures/quality_adjusted_goodput_by_router.png`
- `reports/figures/memory_token_footprint_by_cache.png`
- `reports/figures/prompt_layout_ablation.png`
- `reports/figures/adapter_strategy_frontier.png`
- `reports/figures/concurrency_p95_ttft.png`
- `reports/figures/concurrency_qag.png`
- `reports/figures/concurrency_slo_attainment.png`
- `reports/figures/concurrency_request_throughput.png`

Generated tables:

- `reports/tables/summaries.csv`
- `reports/tables/workload_leaders.csv`
- `reports/tables/cache_model_means.csv`
- `reports/tables/router_means.csv`
- `reports/tables/repeated_seed_summary.csv`
- `reports/tables/layout_ablation.csv`
- `reports/tables/pareto_frontier.csv`
- `reports/tables/slo_sweep.csv`
- `reports/tables/adapter_cache_metrics.csv`

### Workload leaders

| workload | router_policy | cache_model | quality_adjusted_goodput | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- | --- | --- |
| agent_session | oracle | activated_lora | 17.118 | 0.922 | 23.241 |
| controlled_overlap | cache_aware | activated_lora | 1.679 | 0.201 | 887.822 |
| jsonl_eval | cache_aware | activated_lora | 19.389 | 0.910 | 24.454 |
| low_overlap_control | semantic | activated_lora | 8.911 | 0.916 | 72.620 |
| mixed_tasks_same_doc | cache_aware | activated_lora | 1693.550 | 0.915 | 19.255 |
| prompt_layout_ablation | sticky_session | activated_lora | 12.220 | 0.932 | 71.645 |
| shared_doc_qa | sticky_session | copy_on_write | 18.199 | 0.924 | 91.558 |

### Cache-model means

| cache_model | adapter_strategy | quality_adjusted_goodput | quality_adjusted_goodput_per_memory_token | p95_ttft_ms | cache_hit_rate | fragmentation_index | eviction_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_shared | multitask-or-shared-base | 13.940 | 0.000 | 52.145 | 0.844 | 1.000 | nan |
| copy_on_write | copy-on-write-delta | 11.945 | 0.003 | 168.414 | 0.796 | 1.029 | 0.000 |
| activated_lora | activated-late-specialization | 11.480 | 0.012 | 858.350 | 0.779 | 1.001 | 0.000 |
| standard_lora | specialist-adapter | 9.373 | 0.002 | 337.503 | 0.772 | 1.126 | 0.122 |

### Router means

| router_policy | quality_adjusted_goodput | quality_adjusted_goodput_per_memory_token | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- | --- |
| oracle | 13.810 | 0.003 | 0.918 | 71.827 |
| sticky_session | 13.810 | 0.003 | 0.918 | 71.827 |
| cache_aware | 12.662 | 0.015 | 0.518 | 868.615 |
| semantic | 11.336 | 0.003 | 0.898 | 241.675 |
| random | 10.508 | 0.000 | 0.685 | 56.757 |
| multitask | 5.698 | 0.002 | 0.549 | 809.996 |

### Repeated-seed summary

| workload | router_policy | cache_model | run_count | quality_adjusted_goodput_mean | quality_adjusted_goodput_std | p95_ttft_ms_mean |
| --- | --- | --- | --- | --- | --- | --- |
| mixed_tasks_same_doc | cache_aware | activated_lora | 10 | 186.641 | 529.475 | 18.899 |
| mixed_tasks_same_doc | cache_aware | copy_on_write | 4 | 18.172 | 0.293 | 19.765 |
| mixed_tasks_same_doc | oracle | copy_on_write | 4 | 18.172 | 0.293 | 19.765 |
| mixed_tasks_same_doc | semantic | copy_on_write | 4 | 18.172 | 0.293 | 19.765 |
| mixed_tasks_same_doc | sticky_session | copy_on_write | 4 | 18.172 | 0.293 | 19.765 |
| mixed_tasks_same_doc | sticky_session | activated_lora | 4 | 18.138 | 0.374 | 18.365 |
| mixed_tasks_same_doc | semantic | activated_lora | 4 | 18.138 | 0.374 | 18.365 |
| mixed_tasks_same_doc | oracle | activated_lora | 4 | 18.138 | 0.374 | 18.365 |
| shared_doc_qa | cache_aware | copy_on_write | 4 | 17.655 | 0.644 | 87.368 |
| shared_doc_qa | cache_aware | standard_lora | 4 | 17.655 | 0.644 | 87.368 |
| shared_doc_qa | oracle | copy_on_write | 4 | 17.655 | 0.644 | 87.368 |
| shared_doc_qa | oracle | standard_lora | 4 | 17.655 | 0.644 | 87.368 |

### Prompt-layout ablation

| prompt_layout | cache_model | ttft_ms | quality | cached_prompt_tokens |
| --- | --- | --- | --- | --- |
| document_before_instruction | activated_lora | 145.294 | 0.781 | 200.760 |
| document_before_instruction | base_shared | 18.256 | 0.882 | 154.000 |
| document_before_instruction | copy_on_write | 18.312 | 0.907 | 204.286 |
| document_before_instruction | standard_lora | 147.611 | 0.781 | 200.957 |
| instruction_before_document | activated_lora | 201.573 | 0.739 | 7.169 |
| instruction_before_document | base_shared | 69.129 | 0.895 | 7.333 |
| instruction_before_document | copy_on_write | 87.239 | 0.897 | 7.333 |
| instruction_before_document | standard_lora | 204.343 | 0.739 | 7.184 |

### Pareto frontier

| pareto_workload | router_policy | cache_model | mean_quality | p95_ttft_ms | quality_adjusted_goodput |
| --- | --- | --- | --- | --- | --- |
| agent_session | oracle | activated_lora | 0.922 | 23.241 | 17.118 |
| controlled_overlap | cache_aware | activated_lora | 0.195 | 871.995 | 1.551 |
| controlled_overlap | cache_aware | activated_lora | 0.201 | 876.553 | 1.615 |
| controlled_overlap | cache_aware | activated_lora | 0.202 | 886.927 | 1.667 |
| controlled_overlap | cache_aware | activated_lora | 0.203 | 910.269 | 1.634 |
| controlled_overlap | cache_aware | standard_lora | 0.203 | 1342.571 | 1.022 |
| controlled_overlap | cache_aware | activated_lora | 0.204 | 1479.640 | 0.857 |
| controlled_overlap | multitask | activated_lora | 0.215 | 2218.034 | 0.034 |
| jsonl_eval | cache_aware | activated_lora | 0.910 | 24.454 | 19.389 |
| jsonl_eval | cache_aware | activated_lora | 0.919 | 31.665 | 16.433 |
| low_overlap_control | random | copy_on_write | 0.646 | 72.236 | 6.201 |
| low_overlap_control | semantic | activated_lora | 0.916 | 72.620 | 8.911 |

### SLO sweep leaders

| ttft_slo_ms | workload | router_policy | cache_model | quality_adjusted_goodput | requests_under_slo |
| --- | --- | --- | --- | --- | --- |
| 100.000 | mixed_tasks_same_doc | cache_aware | activated_lora | 20.062 | 32 |
| 150.000 | mixed_tasks_same_doc | cache_aware | activated_lora | 20.062 | 32 |
| 250.000 | mixed_tasks_same_doc | cache_aware | activated_lora | 20.062 | 32 |
| 50.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
| 25.000 | mixed_tasks_same_doc | cache_aware | activated_lora | 18.808 | 30 |

## Takeaways

Specialization is most attractive when quality gains exceed the prefill and
memory cost of lost prefix reuse. Cache-aware and late-specialization strategies
recover locality without collapsing every task into one multitask adapter.

## Limitations

The default backend is a simulator. Tokenization, queueing, and quality are
approximate. Real serving behavior should be validated with vLLM or another
production server before making capacity decisions.

## Physical AI analogue

The same structure appears in VLA and robotics serving: repeated scene tokens
map to a world-state cache, skill adapters map to embodiment or task
specialization, and goodput maps to success-rate-adjusted control Hz.
