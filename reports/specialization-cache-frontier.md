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

Best quality-adjusted goodput in the current artifact set is `cache_aware` with `activated_lora` on `mixed_tasks_same_doc`. Mean quality is 0.915, p95 TTFT is 19.3 ms, and fragmentation index is 1.00. `document_before_instruction` mean TTFT is 19.2 ms. `instruction_before_document` mean TTFT is 69.2 ms.

Generated figures:

- `reports/figures/quality_vs_p95_ttft.png`
- `reports/figures/cache_hit_rate_by_policy_model.png`
- `reports/figures/quality_adjusted_goodput_by_router.png`
- `reports/figures/memory_token_footprint_by_cache.png`
- `reports/figures/prompt_layout_ablation.png`
- `reports/figures/adapter_strategy_frontier.png`

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
