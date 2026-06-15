# Large-model vLLM results

Date: 2026-06-15

This page records real vLLM serving runs for a larger base model. These are
base-only systems runs: no large-model specialist or multitask adapters were
trained for this run.

Serving stack:

- GCP `g2-standard-8` with one NVIDIA L4.
- vLLM OpenAI-compatible server, `vllm/vllm-openai:latest`, vLLM `0.23.0`.
- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Context: `max_model_len=4096`.
- Workload: `controlled_overlap`.
- Cache model in benchmark: `activated_lora`.
- Streaming TTFT enabled.
- SLO: `ttft_slo_ms=1500`.

Run command:

```bash
make vllm-large-model-pilot
```

Confidence sweep command:

```bash
make vllm-large-model-confidence-reset
```

## Five-seed isolated confidence sweep

The stronger result uses five seeds and restarts vLLM before every condition so
prefix-cache state cannot leak across overlap levels. Each condition serves 40
streamed requests at concurrency 4, for 400 total requests.

![Qwen2.5-7B overlap confidence sweep](figures/large_model_overlap_confidence.png)

| overlap | runs | requests | p50 TTFT mean ms | p95 TTFT mean ms | p95 TTFT std ms | p99 TTFT mean ms | p95 E2E mean ms | SLO attainment mean | req/s mean | quality mean | QAG mean | cached ratio mean | server prefix hit mean | memory tokens mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 5 | 200 | 1515.5 | 2429.7 | 103.0 | 2544.1 | 4454.6 | 0.465 | 1.081 | 0.067 | 0.034 | 0.408 | 0.264 | 4346 |
| 0.95 | 5 | 200 | 924.4 | 1725.2 | 37.6 | 1762.2 | 3693.3 | 0.900 | 1.363 | 0.073 | 0.090 | 0.778 | 0.838 | 1466 |

High-overlap prompts improved p95 TTFT by `704.5 ms` on average, a `29.0%`
reduction. SLO attainment rose from `46.5%` to `90.0%`, request throughput rose
by `26.0%`, and quality-adjusted goodput rose by `163.8%`.

Interpretation:

- This validates the large-base serving side of the thesis: for a 7B causal
  transformer on one L4, shared-prefix locality can move a workload from
  partially SLO-violating to mostly SLO-compliant.
- The server prefix-cache hit rate tracks the benchmark-side cache model:
  `26.4%` at 50% overlap versus `83.8%` at 95% overlap.
- The low-overlap condition also has a larger simulated memory-token footprint
  because fewer shared blocks are reused in the activated-late-specialization
  cache model.
- This still does not prove the specialist-adapter quality side for 7B. The next
  large-model claim requires trained 7B adapters and held-out evaluation through
  the same vLLM harness.

## Two-condition pilot

The corrected pilot used two controlled-overlap conditions at concurrency 4:

| overlap | requests | p50 TTFT ms | p95 TTFT ms | p95 E2E ms | SLO attainment | req/s | quality | QAG | benchmark cached ratio | server prefix hit rate | memory tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 40 | 1506.7 | 2524.2 | 4528.8 | 0.500 | 1.088 | 0.068 | 0.037 | 0.408 | 0.264 | 4346 |
| 0.95 | 40 | 1008.8 | 1153.2 | 3135.9 | 1.000 | 1.377 | 0.074 | 0.102 | 0.778 | 0.848 | 1466 |

Interpretation:

- High shared-prefix overlap made the 7B run SLO-feasible in this pilot:
  p95 TTFT dropped from `2524.2 ms` at 50% overlap to `1153.2 ms` at 95%
  overlap.
- Server prefix-cache hit rate rose from `26.4%` to `84.8%`.
- Benchmark cached prompt-token ratio rose from `40.8%` to `77.8%`.
- Quality is not the conclusion here because this was a base-only systems run.
  The value is the scaling signal: larger causal transformers make prefix-cache
  locality materially more important to TTFT and goodput.

Notes:

- A separate 7B JSONL serving smoke also completed, but it should not be used as
  an overlap ablation because `jsonl_eval` does not consume
  `shared_prefix_fraction`. The controlled-overlap rows above are the relevant
  large-model cache/SLO evidence.
- The next stronger result requires trained 7B specialist and multitask adapters
  served through vLLM, then the same held-out evaluation through
  `configs/benchmark/model_family_vllm_template.yaml`.
