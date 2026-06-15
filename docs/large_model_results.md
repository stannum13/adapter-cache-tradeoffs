# Large-model pilot results

Date: 2026-06-15

This is a real vLLM serving pilot for a larger base model. It is base-only: no
large-model specialist or multitask adapters were trained for this run.

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
