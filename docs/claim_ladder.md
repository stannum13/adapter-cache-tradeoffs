# Claim ladder

This repository studies a systems question:

> When is adapter specialization worth its KV-cache footprint?

The current evidence supports conditional claims. It does not yet support broad
claims about all adapters, all model families, or production serving stacks.

## Claim 1: cache locality is a first-order serving variable

Status: **supported by reset-isolated vLLM evidence and simulator evidence**.

Measured evidence:

Run provenance: June 16, 2026 rerun on GCP `g2-standard-8` with one NVIDIA L4
in `asia-south1-b`, Qwen2.5-7B, vLLM `0.23.0`, server reset before every
condition.

| condition | model/server | requests | runs | overlap | server prefix hit rate | p95 TTFT | SLO attainment | QAG |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium overlap | Qwen2.5-7B on one L4, vLLM reset per condition | 200 | 5 | 50% | 26.4% | 1603.7 ms | 90.0% | 0.080 |
| high overlap | Qwen2.5-7B on one L4, vLLM reset per condition | 200 | 5 | 95% | 83.8% | 937.7 ms | 100.0% | 0.130 |

Effect size:

- server prefix-cache hit rate increased by `57.4` percentage points;
- mean p95 TTFT fell by `666.0 ms`, a `41.5%` reduction;
- SLO attainment rose by `10.0` percentage points;
- quality-adjusted goodput rose by `62.3%`;
- request throughput rose by `33.6%`.

Best current wording:

> On a reset-isolated Qwen2.5-7B vLLM sweep, shared-prefix locality moved the
> workload from partially SLO-violating to mostly SLO-compliant. Cache locality
> is therefore not a secondary metric; it changes the feasible serving region.

## Claim 2: specialization can buy quality, but the win is workload-bound

Status: **supported on the included source-backed and generated fixtures**.

Measured evidence:

| condition | model/server | requests | mean quality | QAG | server prefix hit rate |
| --- | --- | ---: | ---: | ---: | ---: |
| base | Qwen2.5-7B, expanded source eval, L4-era run | 240 | 0.329 | 0.171 | 55.0% |
| specialist LoRAs | Qwen2.5-7B, expanded source eval, L4-era run | 240 | 0.547 | 0.544 | 47.9% |
| multitask LoRA | Qwen2.5-7B, expanded source eval, L4-era run | 240 | 0.540 | 0.461 | 55.0% |
| base | Qwen2.5-7B, expanded source eval, H100 run | 240 | 0.326 | 0.503 | 55.0% |
| specialist LoRAs | Qwen2.5-7B, expanded source eval, H100 run | 240 | 0.552 | 0.964 | 50.0% |
| multitask LoRA | Qwen2.5-7B, expanded source eval, H100 run | 240 | 0.541 | 0.921 | 55.0% |

Best current wording:

> On the included source-backed eval, trained specialist LoRAs improved task
> quality versus the base causal transformer and slightly beat a multitask LoRA.
> The cache counters also show the expected cost: specialist routing had lower
> server-level prefix-cache hit rate than the base and multitask conditions.

Avoid saying:

- specialists are universally better than multitask adapters;
- these scores are comparable to standard public LLM benchmarks;
- the quality margin will hold on independently curated data.

## Claim 3: adapter count changes serving capacity

Status: **supported by vLLM startup success/failure records**.

Measured evidence:

| GPU | model | context | registered LoRAs | result |
| --- | --- | ---: | ---: | --- |
| NVIDIA L4 24GB | Qwen2.5-7B | 4096 | 5 | starts |
| NVIDIA L4 24GB | Qwen2.5-7B | 4096 | 8 | fails: only 0.09 GiB available KV cache |
| NVIDIA L4 24GB | Qwen2.5-7B | 4096 | 10 | fails: no available memory for cache blocks |
| NVIDIA H100 80GB | Qwen2.5-7B | 4096 | 10 | starts; 53.34 GiB available KV cache |

Best current wording:

> Adapter registration is part of the capacity frontier. On one L4, the same
> 4096-context Qwen2.5-7B deployment shape that could serve five LoRAs failed
> at eight and ten registered LoRAs. On one H100 80GB, the ten-LoRA shape
> started with large KV headroom.

## Claim 4: late specialization is a plausible mitigation, not yet a kernel result

Status: **supported by simulator tests only**.

The simulator shows why activated-LoRA-style invocation can preserve prefix
sharing when the shared document appears before the adapter invocation marker.
It does not prove that a deployed vLLM kernel has this behavior.

Best current wording:

> Activated-LoRA-style late specialization is modeled as a cache-compatible base
> prefix followed by adapter-specific tokens. The simulator shows the mechanism;
> production validation requires a serving stack with the corresponding kernel
> or cache-key behavior.

## Decision rule

Treat specialization as worth it only when:

```text
quality gain
  >
latency SLO loss + cache-memory footprint + tenant-isolation cost
```

Operationally, this repo measures that with:

- `mean_quality` for task correctness;
- `p95_ttft_ms`, `slo_attainment_rate`, and `quality_adjusted_goodput` for
  user-visible serving value;
- `server_prefix_cache_hit_rate` for vLLM-level cache reuse;
- `cached_prompt_token_ratio`, `fragmentation_index`, and
  `memory_token_footprint` for benchmark-side cache accounting;
- capacity records for startup feasibility under model, context, and adapter
  count.

## What remains unproven

- independently curated external eval performance;
- multi-family results beyond the current Qwen-centered evidence;
- adapter-aware vLLM cache counters rather than server-level counters;
- production activated-LoRA cache-key behavior;
- dollar-cost and power-cost frontiers across GPU types.
