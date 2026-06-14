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

Best quality-adjusted goodput in the current artifact set is `cache_aware` with `activated_lora` on `jsonl_eval`. Mean quality is 0.910, p95 TTFT is 24.5 ms, and fragmentation index is 1.00. `document_before_instruction` mean TTFT is 42.8 ms. `instruction_before_document` mean TTFT is 73.2 ms.

### Interpretation

- Best aggregate cache strategy: `activated_lora` with mean quality-adjusted goodput 14.251.
- Prompt layout matters: `document_before_instruction` is 29.3 ms lower mean TTFT than `instruction_before_document` in the current artifact set.
- Highest eviction pressure: `standard_lora` on `mixed_tasks_same_doc` with 1551 evictions.
- Repeated-seed leader: `cache_aware` with `base_shared` on `mixed_tasks_same_doc`.

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
- `reports/tables/repeated_seed_summary.csv`
- `reports/tables/layout_ablation.csv`
- `reports/tables/pareto_frontier.csv`
- `reports/tables/slo_sweep.csv`

### Workload leaders

| workload | router_policy | cache_model | quality_adjusted_goodput | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- | --- | --- |
| agent_session | oracle | activated_lora | 17.118 | 0.922 | 23.241 |
| jsonl_eval | cache_aware | activated_lora | 19.389 | 0.910 | 24.454 |
| low_overlap_control | oracle | standard_lora | 8.911 | 0.916 | 72.620 |
| mixed_tasks_same_doc | oracle | copy_on_write | 18.892 | 0.912 | 17.849 |
| prompt_layout_ablation | oracle | activated_lora | 12.337 | 0.930 | 71.699 |
| shared_doc_qa | semantic | activated_lora | 18.502 | 0.922 | 70.629 |

### Cache-model means

| cache_model | adapter_strategy | quality_adjusted_goodput | p95_ttft_ms | cache_hit_rate | fragmentation_index | eviction_count |
| --- | --- | --- | --- | --- | --- | --- |
| activated_lora | activated-late-specialization | 14.251 | 54.487 | 0.748 | 0.842 | 263.155 |
| copy_on_write | copy-on-write-delta | 14.144 | 55.023 | 0.745 | 0.858 | 268.543 |
| base_shared | multitask-or-shared-base | 13.025 | 55.767 | 0.625 | 0.742 | 437.163 |
| standard_lora | specialist-adapter | 12.763 | 68.162 | 0.687 | 1.000 | 353.514 |

### Router means

| router_policy | quality_adjusted_goodput | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- |
| sticky_session | 14.560 | 0.920 | 56.178 |
| cache_aware | 13.807 | 0.896 | 55.191 |
| oracle | 13.802 | 0.920 | 60.965 |
| semantic | 13.802 | 0.920 | 60.965 |
| random | 10.508 | 0.685 | 56.757 |

### Repeated-seed summary

| workload | router_policy | cache_model | run_count | quality_adjusted_goodput_mean | quality_adjusted_goodput_std | p95_ttft_ms_mean |
| --- | --- | --- | --- | --- | --- | --- |
| mixed_tasks_same_doc | cache_aware | base_shared | 3 | 18.685 | 0.208 | 18.951 |
| mixed_tasks_same_doc | oracle | base_shared | 3 | 18.685 | 0.208 | 18.951 |
| mixed_tasks_same_doc | semantic | base_shared | 3 | 18.685 | 0.208 | 18.951 |
| mixed_tasks_same_doc | oracle | activated_lora | 6 | 18.386 | 0.340 | 18.260 |
| mixed_tasks_same_doc | semantic | activated_lora | 6 | 18.386 | 0.340 | 18.260 |
| mixed_tasks_same_doc | cache_aware | activated_lora | 6 | 18.386 | 0.340 | 18.260 |
| mixed_tasks_same_doc | oracle | copy_on_write | 6 | 18.340 | 0.427 | 18.878 |
| mixed_tasks_same_doc | cache_aware | copy_on_write | 6 | 18.340 | 0.427 | 18.878 |
| mixed_tasks_same_doc | semantic | copy_on_write | 6 | 18.340 | 0.427 | 18.878 |
| shared_doc_qa | cache_aware | activated_lora | 6 | 14.548 | 4.910 | 75.120 |
| shared_doc_qa | oracle | activated_lora | 6 | 14.548 | 4.910 | 75.120 |
| shared_doc_qa | semantic | activated_lora | 6 | 14.548 | 4.910 | 75.120 |

### Prompt-layout ablation

| prompt_layout | cache_model | ttft_ms | quality | cached_prompt_tokens |
| --- | --- | --- | --- | --- |
| document_before_instruction | activated_lora | 39.880 | 0.926 | 98.462 |
| document_before_instruction | base_shared | 55.444 | 0.916 | 59.231 |
| document_before_instruction | copy_on_write | 40.042 | 0.926 | 98.000 |
| document_before_instruction | standard_lora | 40.933 | 0.926 | 95.455 |
| instruction_before_document | activated_lora | 72.848 | 0.912 | 4.561 |
| instruction_before_document | base_shared | 75.082 | 0.909 | 2.821 |
| instruction_before_document | copy_on_write | 72.810 | 0.912 | 4.667 |
| instruction_before_document | standard_lora | 72.848 | 0.912 | 4.561 |

### Pareto frontier

| pareto_workload | router_policy | cache_model | mean_quality | p95_ttft_ms | quality_adjusted_goodput |
| --- | --- | --- | --- | --- | --- |
| agent_session | oracle | activated_lora | 0.922 | 23.241 | 17.118 |
| jsonl_eval | cache_aware | activated_lora | 0.910 | 24.454 | 19.389 |
| low_overlap_control | random | copy_on_write | 0.646 | 72.236 | 6.201 |
| low_overlap_control | oracle | standard_lora | 0.916 | 72.620 | 8.911 |
| mixed_tasks_same_doc | semantic | activated_lora | 0.912 | 17.296 | 18.786 |
| mixed_tasks_same_doc | oracle | activated_lora | 0.914 | 19.255 | 18.603 |
| prompt_layout_ablation | cache_aware | activated_lora | 0.932 | 71.645 | 12.220 |
| shared_doc_qa | cache_aware | copy_on_write | 0.922 | 70.629 | 18.384 |
| shared_doc_qa | cache_aware | copy_on_write | 0.924 | 71.111 | 17.292 |

### SLO sweep leaders

| ttft_slo_ms | workload | router_policy | cache_model | quality_adjusted_goodput | requests_under_slo |
| --- | --- | --- | --- | --- | --- |
| 250.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
| 150.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
| 100.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
| 50.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
| 25.000 | jsonl_eval | cache_aware | activated_lora | 18.419 | 95 |

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
