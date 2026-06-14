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

The first implementation uses a deterministic mock backend, whitespace
tokenization, block prefix caching, and synthetic quality matrices.
Real vLLM integration is intentionally optional.

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

Best quality-adjusted goodput in the current artifact set is `cache_aware` with `activated_lora` on `mixed_tasks_same_doc`. Mean quality is 0.914, p95 TTFT is 19.3 ms, and fragmentation index is 1.00. `document_before_instruction` mean TTFT is 19.2 ms. `instruction_before_document` mean TTFT is 69.2 ms.

Generated figures:

- `reports/figures/quality_vs_p95_ttft.png`
- `reports/figures/cache_hit_rate_by_policy_model.png`
- `reports/figures/quality_adjusted_goodput_by_router.png`
- `reports/figures/memory_token_footprint_by_cache.png`
- `reports/figures/prompt_layout_ablation.png`
- `reports/figures/adapter_strategy_frontier.png`

Generated tables:

- `reports/tables/summaries.csv`
- `reports/tables/workload_leaders.csv`
- `reports/tables/cache_model_means.csv`
- `reports/tables/router_means.csv`
- `reports/tables/layout_ablation.csv`
- `reports/tables/pareto_frontier.csv`
- `reports/tables/slo_sweep.csv`

### Workload leaders

| workload | router_policy | cache_model | quality_adjusted_goodput | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- | --- | --- |
| agent_session | oracle | activated_lora | 17.118 | 0.922 | 23.241 |
| low_overlap_control | oracle | standard_lora | 8.911 | 0.916 | 72.620 |
| mixed_tasks_same_doc | oracle | activated_lora | 18.603 | 0.914 | 19.255 |
| prompt_layout_ablation | cache_aware | activated_lora | 12.220 | 0.932 | 71.645 |
| shared_doc_qa | oracle | activated_lora | 16.822 | 0.913 | 73.764 |

### Cache-model means

| cache_model | adapter_strategy | quality_adjusted_goodput | p95_ttft_ms | cache_hit_rate | fragmentation_index | eviction_count |
| --- | --- | --- | --- | --- | --- | --- |
| activated_lora | activated-late-specialization | 13.967 | 52.249 | 0.841 | 1.018 | 0.000 |
| base_shared | multitask-or-shared-base | 13.940 | 52.145 | 0.844 | 1.000 | 0.000 |
| copy_on_write | copy-on-write-delta | 13.940 | 52.145 | 0.844 | 1.026 | 0.000 |
| standard_lora | specialist-adapter | 12.863 | 65.348 | 0.834 | 1.347 | 0.000 |

### Router means

| router_policy | quality_adjusted_goodput | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- |
| oracle | 14.560 | 0.920 | 56.178 |
| semantic | 14.560 | 0.920 | 56.178 |
| sticky_session | 14.560 | 0.920 | 56.178 |
| cache_aware | 14.200 | 0.893 | 52.068 |
| random | 10.508 | 0.685 | 56.757 |

### Prompt-layout ablation

| prompt_layout | cache_model | ttft_ms | quality | cached_prompt_tokens |
| --- | --- | --- | --- | --- |
| document_before_instruction | activated_lora | 18.122 | 0.882 | 154.383 |
| document_before_instruction | base_shared | 18.256 | 0.882 | 154.000 |
| document_before_instruction | copy_on_write | 18.256 | 0.882 | 154.000 |
| document_before_instruction | standard_lora | 22.176 | 0.882 | 142.800 |
| instruction_before_document | activated_lora | 69.293 | 0.895 | 6.867 |
| instruction_before_document | base_shared | 69.129 | 0.895 | 7.333 |
| instruction_before_document | copy_on_write | 69.129 | 0.895 | 7.333 |
| instruction_before_document | standard_lora | 69.293 | 0.895 | 6.867 |

### Pareto frontier

| pareto_workload | router_policy | cache_model | mean_quality | p95_ttft_ms | quality_adjusted_goodput |
| --- | --- | --- | --- | --- | --- |
| agent_session | oracle | activated_lora | 0.922 | 23.241 | 17.118 |
| low_overlap_control | random | copy_on_write | 0.646 | 72.236 | 6.201 |
| low_overlap_control | oracle | standard_lora | 0.916 | 72.620 | 8.911 |
| mixed_tasks_same_doc | cache_aware | standard_lora | 0.690 | 17.731 | 13.988 |
| mixed_tasks_same_doc | oracle | activated_lora | 0.914 | 19.255 | 18.603 |
| prompt_layout_ablation | cache_aware | activated_lora | 0.932 | 71.645 | 12.220 |
| shared_doc_qa | random | copy_on_write | 0.709 | 72.749 | 13.379 |
| shared_doc_qa | oracle | activated_lora | 0.913 | 73.764 | 16.822 |

### SLO sweep leaders

| ttft_slo_ms | workload | router_policy | cache_model | quality_adjusted_goodput | requests_under_slo |
| --- | --- | --- | --- | --- | --- |
| 250.000 | mixed_tasks_same_doc | oracle | activated_lora | 18.603 | 48 |
| 100.000 | mixed_tasks_same_doc | semantic | activated_lora | 18.603 | 48 |
| 150.000 | mixed_tasks_same_doc | oracle | activated_lora | 18.603 | 48 |
| 50.000 | mixed_tasks_same_doc | sticky_session | activated_lora | 17.828 | 46 |
| 25.000 | mixed_tasks_same_doc | sticky_session | activated_lora | 17.828 | 46 |

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
