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

The committed report uses the deterministic mock backend as a systems
simulator: it isolates routing, prefix-cache locality, finite KV budget,
eviction, prompt layout, and SLO behavior from GPU and serving noise.
It is not model-quality evidence. Real model evaluation should use the
same JSONL workloads with `backend.kind: vllm`.

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

Best quality-adjusted goodput in the current artifact set is `cache_aware` with `activated_lora` on `jsonl_eval`. Mean quality is 0.910, p95 TTFT is 24.5 ms, and fragmentation index is 1.00. `document_before_instruction` mean TTFT is 43.2 ms. `instruction_before_document` mean TTFT is 73.3 ms.

### Interpretation

- Best aggregate cache strategy: `activated_lora` with mean quality-adjusted goodput 13.918.
- Prompt layout matters: `document_before_instruction` is 28.9 ms lower mean TTFT than `instruction_before_document` in the current artifact set.
- Highest eviction pressure: `standard_lora` on `mixed_tasks_same_doc` with 1551 evictions.
- Repeated-seed leader: `semantic` with `base_shared` on `mixed_tasks_same_doc`.

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
| agent_session | oracle | activated_lora | 16.632 | 0.923 | 23.240 |
| jsonl_eval | cache_aware | activated_lora | 19.389 | 0.910 | 24.454 |
| low_overlap_control | cache_aware | activated_lora | 8.840 | 0.907 | 72.239 |
| mixed_tasks_same_doc | cache_aware | base_shared | 18.892 | 0.912 | 17.849 |
| prompt_layout_ablation | cache_aware | activated_lora | 12.337 | 0.930 | 71.699 |
| shared_doc_qa | oracle | activated_lora | 18.502 | 0.922 | 70.629 |

### Cache-model means

| cache_model | adapter_strategy | quality_adjusted_goodput | p95_ttft_ms | cache_hit_rate | fragmentation_index | eviction_count |
| --- | --- | --- | --- | --- | --- | --- |
| activated_lora | activated-late-specialization | 13.918 | 54.574 | 0.744 | 0.833 | 273.758 |
| copy_on_write | copy-on-write-delta | 13.822 | 55.023 | 0.742 | 0.851 | 278.489 |
| standard_lora | specialist-adapter | 12.760 | 65.391 | 0.696 | 0.956 | 344.578 |
| base_shared | multitask-or-shared-base | 12.694 | 55.645 | 0.611 | 0.726 | 464.148 |

### Router means

| router_policy | quality_adjusted_goodput | mean_quality | p95_ttft_ms |
| --- | --- | --- | --- |
| sticky_session | 14.571 | 0.917 | 55.494 |
| cache_aware | 13.810 | 0.896 | 55.017 |
| oracle | 13.805 | 0.919 | 60.773 |
| semantic | 13.805 | 0.919 | 60.773 |
| multitask | 12.647 | 0.811 | 55.960 |
| random | 10.022 | 0.644 | 56.712 |

### Repeated-seed summary

| workload | router_policy | cache_model | run_count | quality_adjusted_goodput_mean | quality_adjusted_goodput_std | p95_ttft_ms_mean |
| --- | --- | --- | --- | --- | --- | --- |
| mixed_tasks_same_doc | semantic | base_shared | 3 | 18.590 | 0.360 | 18.649 |
| mixed_tasks_same_doc | oracle | base_shared | 3 | 18.590 | 0.360 | 18.649 |
| mixed_tasks_same_doc | cache_aware | base_shared | 3 | 18.590 | 0.360 | 18.649 |
| mixed_tasks_same_doc | cache_aware | activated_lora | 6 | 18.338 | 0.323 | 18.109 |
| mixed_tasks_same_doc | semantic | activated_lora | 6 | 18.338 | 0.323 | 18.109 |
| mixed_tasks_same_doc | oracle | activated_lora | 6 | 18.338 | 0.323 | 18.109 |
| mixed_tasks_same_doc | semantic | copy_on_write | 6 | 18.293 | 0.424 | 18.727 |
| mixed_tasks_same_doc | cache_aware | copy_on_write | 6 | 18.293 | 0.424 | 18.727 |
| mixed_tasks_same_doc | oracle | copy_on_write | 6 | 18.293 | 0.424 | 18.727 |
| mixed_tasks_same_doc | multitask | base_shared | 3 | 16.620 | 0.524 | 19.475 |
| mixed_tasks_same_doc | multitask | activated_lora | 6 | 16.597 | 0.554 | 18.973 |
| mixed_tasks_same_doc | multitask | standard_lora | 6 | 16.551 | 0.534 | 19.556 |

### Prompt-layout ablation

| prompt_layout | cache_model | ttft_ms | quality | cached_prompt_tokens |
| --- | --- | --- | --- | --- |
| document_before_instruction | activated_lora | 40.222 | 0.896 | 97.192 |
| document_before_instruction | base_shared | 56.667 | 0.890 | 55.440 |
| document_before_instruction | copy_on_write | 40.395 | 0.896 | 96.698 |
| document_before_instruction | standard_lora | 40.993 | 0.896 | 94.988 |
| instruction_before_document | activated_lora | 72.867 | 0.886 | 4.523 |
| instruction_before_document | base_shared | 75.306 | 0.881 | 2.640 |
| instruction_before_document | copy_on_write | 72.839 | 0.886 | 4.605 |
| instruction_before_document | standard_lora | 72.867 | 0.886 | 4.523 |

### Pareto frontier

| pareto_workload | router_policy | cache_model | mean_quality | p95_ttft_ms | quality_adjusted_goodput |
| --- | --- | --- | --- | --- | --- |
| agent_session | multitask | base_shared | 0.816 | 23.011 | 15.079 |
| agent_session | oracle | activated_lora | 0.923 | 23.240 | 16.632 |
| jsonl_eval | cache_aware | activated_lora | 0.910 | 24.454 | 19.389 |
| low_overlap_control | multitask | activated_lora | 0.803 | 71.517 | 7.838 |
| low_overlap_control | cache_aware | activated_lora | 0.907 | 72.239 | 8.840 |
| mixed_tasks_same_doc | multitask | activated_lora | 0.802 | 17.127 | 16.902 |
| mixed_tasks_same_doc | oracle | activated_lora | 0.912 | 17.296 | 18.786 |
| prompt_layout_ablation | multitask | copy_on_write | 0.812 | 71.447 | 10.539 |
| prompt_layout_ablation | oracle | copy_on_write | 0.930 | 71.699 | 12.311 |
| shared_doc_qa | cache_aware | standard_lora | 0.922 | 70.629 | 18.384 |
| shared_doc_qa | oracle | standard_lora | 0.924 | 71.111 | 17.292 |

### SLO sweep leaders

| ttft_slo_ms | workload | router_policy | cache_model | quality_adjusted_goodput | requests_under_slo |
| --- | --- | --- | --- | --- | --- |
| 100.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
| 150.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
| 250.000 | jsonl_eval | cache_aware | activated_lora | 19.389 | 100 |
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
