# When is specialization worth its cache footprint?

## Abstract

Specialist adapters can improve task quality, but every adapter routing decision
also changes the cache namespace seen by a causal transformer serving stack.
This benchmark studies that tradeoff under shared-prefix workloads.

## Core question

When is model or adapter specialization worth its KV-cache footprint?

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

- `mock` / `mock-causal-transformer`: 902 runs, 103952 requests, mean quality 0.902, mean p95 TTFT 83.0 ms.
- `legacy/unclassified` / `provenance unavailable`: 116 runs, 5221 requests, mean quality 0.837, mean p95 TTFT 111.4 ms. These rows are listed for completeness, not as claim-supporting evidence.
- `vllm` / `Qwen/Qwen2.5-1.5B-Instruct`: 199 runs, 15036 requests, mean quality 0.448, mean p95 TTFT 1228.5 ms.
- `vllm` / `Qwen/Qwen2.5-7B-Instruct`: 46 runs, 3432 requests, mean quality 0.274, mean p95 TTFT 1033.7 ms.
- `vllm` / `TinyLlama/TinyLlama-1.1B-Chat-v1.0`: 6 runs, 3000 requests, mean quality 0.375, mean p95 TTFT 87.4 ms.

### Claim ladder

The maintained public claim boundary lives in
[docs/claim_ladder.md](claim_ladder.md). Current supported claims
must cite model/server, request count, run count, and metric scope.

- Reset-isolated 7B cache locality: 400 requests across 10 runs. Moving from 50% to 95% shared-prefix overlap raised server prefix-cache hit rate by 57.4 percentage points, reduced mean p95 TTFT by 666.0 ms (41.5%), lifted SLO attainment by 10.0 percentage points, and raised QAG by 62.3%.

### Claim boundary

The report separates simulator-backed findings from real-serving claims.
Treat missing gates as scope limits, not as negative results.

| Claim area | Status | Evidence in this report | Required before widening |
| --- | --- | --- | --- |
| Simulator regime map | supported in this report | mock `regime_*` runs with structure metrics and policy regret | reset-isolated real-server bridge |
| Cache-control mechanisms | simulator-backed | `warm`, `cold`, and `prefix_disabled` rows | server reset settings and cache-counter provenance |
| Real-server regime bridge | not supported here | no reset-isolated vLLM regime sweep in this artifact set | repeat claim-critical regimes with comparable conditions |
| Prefix-cache causality | not established here | no positive server-side prefix/cache counters for `regime_*` vLLM bridge rows | capture counters or downgrade to client-observed behavior |
| Automated recommendations | deferred | policy comparisons are explanatory, not prescriptive | G8 bridge plus uncertainty and user-path readiness |

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

Exploratory aggregate leader in the loaded artifact summaries is `cache_aware` with `activated_lora` on `mixed_tasks_same_doc`. Mean quality is 0.915, p95 TTFT is 19.3 ms, and fragmentation index is 1.00. `document_before_instruction` mean TTFT is 105.4 ms. `instruction_before_document` mean TTFT is 165.2 ms. This is not a public benchmark claim; use the claim ladder for supported evidence boundaries.

### Decision rule

Treat specialization as worthwhile only when it improves quality-adjusted
goodput under the TTFT SLO after accounting for cache memory. In this
repo that means comparing both `quality_adjusted_goodput` and
`quality_adjusted_goodput_per_memory_token`, while checking that the
fragmentation index and SLO attainment do not regress beyond the serving
budget for the workload.

### Interpretation

- Best aggregate cache strategy: `base_shared` with mean quality-adjusted goodput 13.940.
- Best comparable cache-footprint efficiency: `activated_lora` with mean quality-adjusted goodput per memory token 0.039240 under `cold` cache conditions.
- Prompt layout matters: `document_before_instruction` is 58.2 ms lower mean TTFT than `instruction_before_document` in the loaded artifact summaries.
- Highest eviction pressure: `standard_lora` on `mixed_tasks_same_doc` with 12 evictions.
- Repeated-seed leader: `cache_aware` with `activated_lora` on `mixed_tasks_same_doc`.

Generated figure artifact paths:

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
- `reports/figures/regime_policy_failure_map.png`

Generated table artifact paths:

- `reports/tables/summaries.csv`
- `reports/tables/workload_leaders.csv`
- `reports/tables/cache_model_means.csv`
- `reports/tables/router_means.csv`
- `reports/tables/repeated_seed_summary.csv`
- `reports/tables/layout_ablation.csv`
- `reports/tables/pareto_frontier.csv`
- `reports/tables/slo_sweep.csv`
- `reports/tables/adapter_cache_metrics.csv`
- `reports/tables/policy_regret.csv`
- `reports/tables/claim_evidence.csv`

### Workload leaders

| workload | cache_condition | router_policy | cache_model | quality_adjusted_goodput | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- | --- | --- | --- |
| agent_session | warm | oracle | activated_lora | 17.118 | 0.922 | 23.241 |
| controlled_overlap | warm | cache_aware | activated_lora | 1.679 | 0.201 | 887.822 |
| jsonl_eval | warm | cache_aware | activated_lora | 19.389 | 0.910 | 24.454 |
| low_overlap_control | warm | semantic | activated_lora | 8.911 | 0.916 | 72.620 |
| mixed_tasks_same_doc | warm | cache_aware | activated_lora | 1693.550 | 0.915 | 19.255 |
| prompt_layout_ablation | warm | sticky_session | activated_lora | 12.220 | 0.932 | 71.645 |
| regime_adversarial_churn | warm | cache_aware | standard_lora | 7.920 | 0.914 | 85.075 |
| regime_adversarial_churn | cold | cache_aware | standard_lora | 7.832 | 0.914 | 86.013 |

### Cache-model means

| cache_model | cache_condition | adapter_strategy | quality_adjusted_goodput | quality_adjusted_goodput_per_memory_token | p95_ttft_ms | cache_hit_rate | fragmentation_index | eviction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_shared | warm | multitask-or-shared-base | 13.940 | 0.000 | 52.145 | 0.844 | 1.000 |  |
| copy_on_write | warm | copy-on-write-delta | 12.282 | 0.002 | 121.615 | 0.809 | 1.017 | 0.000 |
| activated_lora | warm | activated-late-specialization | 11.849 | 0.009 | 638.615 | 0.791 | 1.001 | 0.000 |
| standard_lora | warm | specialist-adapter | 9.602 | 0.001 | 212.944 | 0.782 | 1.345 | 0.055 |
| activated_lora | cold | activated-late-specialization | 7.926 | 0.039 | 84.825 | 0.000 | 1.000 | 0.000 |
| activated_lora | prefix_disabled | activated-late-specialization | 7.926 | 7.926 | 84.825 | 0.000 | 0.000 | 0.000 |
| copy_on_write | cold | copy-on-write-delta | 7.926 | 0.038 | 84.825 | 0.000 | 1.044 | 0.000 |
| copy_on_write | prefix_disabled | copy-on-write-delta | 7.926 | 7.926 | 84.825 | 0.000 | 0.000 | 0.000 |

### Router means

| router_policy | cache_condition | quality_adjusted_goodput | quality_adjusted_goodput_per_memory_token | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- | --- | --- |
| oracle | warm | 12.536 | 0.002 | 0.914 | 79.321 |
| sticky_session | warm | 12.536 | 0.002 | 0.914 | 79.321 |
| cache_aware | warm | 12.383 | 0.011 | 0.611 | 648.570 |
| semantic | warm | 11.563 | 0.002 | 0.906 | 154.069 |
| random | warm | 10.508 | 0.000 | 0.685 | 56.757 |
| cache_aware | cold | 7.926 | 0.039 | 0.912 | 84.825 |
| cache_aware | prefix_disabled | 7.926 | 7.926 | 0.912 | 84.825 |
| oracle | cold | 7.926 | 0.039 | 0.912 | 84.825 |

### Repeated-seed summary

| workload | cache_condition | router_policy | cache_model | run_count | quality_adjusted_goodput_mean | quality_adjusted_goodput_std | p95_ttft_ms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixed_tasks_same_doc | warm | cache_aware | activated_lora | 10 | 186.641 | 529.475 | 18.899 |
| mixed_tasks_same_doc | warm | cache_aware | copy_on_write | 4 | 18.172 | 0.293 | 19.765 |
| mixed_tasks_same_doc | warm | oracle | copy_on_write | 4 | 18.172 | 0.293 | 19.765 |
| mixed_tasks_same_doc | warm | semantic | copy_on_write | 4 | 18.172 | 0.293 | 19.765 |
| mixed_tasks_same_doc | warm | sticky_session | copy_on_write | 4 | 18.172 | 0.293 | 19.765 |
| mixed_tasks_same_doc | warm | oracle | activated_lora | 4 | 18.138 | 0.374 | 18.365 |
| mixed_tasks_same_doc | warm | semantic | activated_lora | 4 | 18.138 | 0.374 | 18.365 |
| mixed_tasks_same_doc | warm | sticky_session | activated_lora | 4 | 18.138 | 0.374 | 18.365 |
| shared_doc_qa | warm | sticky_session | standard_lora | 4 | 17.655 | 0.644 | 87.368 |
| shared_doc_qa | warm | cache_aware | copy_on_write | 4 | 17.655 | 0.644 | 87.368 |
| shared_doc_qa | warm | cache_aware | standard_lora | 4 | 17.655 | 0.644 | 87.368 |
| shared_doc_qa | warm | oracle | copy_on_write | 4 | 17.655 | 0.644 | 87.368 |

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
| 250.000 | mixed_tasks_same_doc | cache_aware | activated_lora | 20.062 | 32 |
| 150.000 | mixed_tasks_same_doc | cache_aware | activated_lora | 20.062 | 32 |
| 100.000 | mixed_tasks_same_doc | cache_aware | activated_lora | 20.062 | 32 |
| 50.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
| 25.000 | mixed_tasks_same_doc | cache_aware | activated_lora | 18.808 | 30 |

## Takeaways

Specialization is most attractive when quality gains exceed the prefill and
memory cost of lost prefix reuse. In simulator runs, cache-aware and
late-specialization-style strategies can recover locality without collapsing
every task into one multitask adapter.

## Limitations

The default backend is a simulator. Tokenization, queueing, and quality are
approximate. Real serving behavior should be validated with vLLM or another
production server before making capacity decisions.

## Physical AI analogue

The same structure appears in VLA and robotics serving: repeated scene tokens
map to a world-state cache, skill adapters map to embodiment or task
specialization, and goodput maps to success-rate-adjusted control Hz.
