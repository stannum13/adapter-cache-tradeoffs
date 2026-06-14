# Real vLLM eval snapshot

This is a reproducible snapshot from a real local vLLM server, not the mock
backend.

Date: 2026-06-15

Serving stack:

- GCP `g2-standard-8` with one NVIDIA L4.
- vLLM OpenAI-compatible server, `vllm/vllm-openai:latest`.
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`.
- LoRA modules: `qa-lora`, `json-lora`, `summary-lora`, `code-lora`,
  `multitask-lora`.
- Dataset: generated public-domain-style JSONL fixtures with held-out request
  rows from `artifacts/sft/public_domain_large` and
  `artifacts/sft/public_domain_xlarge`.
- Metrics: each run saved before/after `/metrics` Prometheus snapshots under
  its ignored artifact directory.

Commands used:

```bash
make vllm-heldout-xlarge-qwen15b
make vllm-heldout-xlarge-lora-trained-qwen15b
make vllm-heldout-xlarge-lora-multitask-qwen15b
make vllm-heldout-trained-matrix-qwen15b
make vllm-heldout-trained-repeated-qwen15b
make vllm-heldout-xlarge-qwen15b-concurrent
make vllm-heldout-xlarge-lora-trained-qwen15b-concurrent
make vllm-heldout-xlarge-lora-multitask-qwen15b-concurrent
```

## What ran

- 39 new vLLM-backed benchmark runs.
- 1,020 real model-server requests.
- 39 saved `backend_metrics_after.prom` snapshots.
- 3 additional concurrent-load vLLM runs at `max_concurrency: 8`.
- 18-run router/cache matrix:
  - routers: `semantic`, `multitask`, `cache_aware`
  - caches: `standard_lora`, `activated_lora`, `copy_on_write`
  - seeds: `17`, `23`
- 18-run repeated-seed matrix:
  - routers: `semantic`, `cache_aware`
  - caches: `standard_lora`, `activated_lora`, `copy_on_write`
  - seeds: `17`, `23`, `31`

## Headline xlarge result

| condition | quality | p95 TTFT ms | QAG | QAG / memory token |
| --- | ---: | ---: | ---: | ---: |
| base model | 0.218 | 1161.3 | 0.169 | 0.000059 |
| trained specialists | 0.851 | 900.9 | 1.592 | 0.000559 |
| multitask adapter | 0.729 | 821.5 | 1.385 | 0.000486 |

On this fixture, specialization clearly improves quality and
quality-adjusted-goodput over the base model. The trained specialists also beat
the multitask adapter on quality and QAG, while the multitask adapter has lower
p95 TTFT.

## Concurrent-load result

These runs used the same 100-request xlarge held-out split with
`backend.max_concurrency: 8`. The concurrent runner reports throughput and
goodput using wall-clock run duration.

| condition | quality | p95 TTFT ms | SLO attainment | request/s | QAG |
| --- | ---: | ---: | ---: | ---: | ---: |
| base model | 0.209 | 2176.1 | 0.030 | 4.687 | 0.029 |
| trained specialists | 0.853 | 1880.6 | 0.310 | 6.172 | 1.632 |
| multitask adapter | 0.727 | 4142.0 | 0.280 | 5.593 | 1.139 |

This strengthens and sharpens the claim:

- Specialist adapters still win on quality and quality-adjusted goodput under
  concurrent load.
- The one-second TTFT SLO is not satisfied at concurrency 8 for this server and
  workload; all three p95 TTFT values exceed it.
- The useful research question becomes an SLO frontier, not a binary yes/no:
  find the concurrency and cache-footprint region where specialization remains
  worth it.

## Concurrency frontier sweep

The overnight frontier config expands `base`, `specialists`, and `multitask`
over `max_concurrency` values `1, 2, 4, 8, 16`:

```bash
make vllm-overnight-frontier
```

| strategy | concurrency | quality | p95 TTFT ms | SLO attainment | request/s | QAG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1 | 0.218 | 1759.6 | 0.080 | 0.721 | 0.013 |
| base | 2 | 0.217 | 2031.4 | 0.010 | 1.308 | 0.003 |
| base | 4 | 0.217 | 2014.0 | 0.030 | 2.626 | 0.017 |
| base | 8 | 0.214 | 2055.9 | 0.080 | 5.243 | 0.090 |
| base | 16 | 0.211 | 3442.1 | 0.010 | 6.906 | 0.015 |
| specialists | 1 | 0.850 | 1588.8 | 0.320 | 0.886 | 0.241 |
| specialists | 2 | 0.856 | 1522.2 | 0.060 | 1.612 | 0.083 |
| specialists | 4 | 0.856 | 1530.4 | 0.110 | 3.258 | 0.307 |
| specialists | 8 | 0.856 | 2043.9 | 0.270 | 6.312 | 1.459 |
| specialists | 16 | 0.850 | 2040.7 | 0.480 | 12.295 | 5.019 |
| multitask | 1 | 0.723 | 1402.8 | 0.310 | 0.894 | 0.200 |
| multitask | 2 | 0.727 | 1741.1 | 0.010 | 1.513 | 0.011 |
| multitask | 4 | 0.730 | 1530.9 | 0.060 | 3.180 | 0.139 |
| multitask | 8 | 0.725 | 1392.7 | 0.350 | 7.009 | 1.778 |
| multitask | 16 | 0.722 | 1535.4 | 0.480 | 12.980 | 4.498 |

Interpretation:

- Specialists dominate base quality and QAG at every tested concurrency.
- Specialists beat multitask at low concurrency and at concurrency 16, while
  multitask leads raw QAG at concurrency 8 in this run.
- None of the strategies satisfy a 1s p95 TTFT SLO on the client-side
  non-streaming latency proxy. The next measurement improvement is streaming
  TTFT or server-side TTFT export per request.
- The generated report now includes concurrency plots for p95 TTFT, QAG, SLO
  attainment, and request throughput.

## Router/cache matrix means

| router | cache | runs | quality | p95 TTFT ms | QAG | QAG / memory token | memory tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic | standard_lora | 5 | 0.820 | 880.4 | 1.624 | 0.001900 | 855 |
| semantic | copy_on_write | 5 | 0.822 | 870.8 | 1.624 | 0.001775 | 915 |
| cache_aware | standard_lora | 5 | 0.820 | 876.2 | 1.614 | 0.001888 | 855 |
| cache_aware | copy_on_write | 5 | 0.822 | 873.8 | 1.584 | 0.001731 | 915 |
| cache_aware | activated_lora | 5 | 0.822 | 890.9 | 1.559 | 0.002192 | 711 |
| semantic | activated_lora | 5 | 0.820 | 912.8 | 1.551 | 0.002181 | 711 |
| multitask | activated_lora | 2 | 0.716 | 880.8 | 1.371 | 0.001929 | 711 |
| multitask | standard_lora | 2 | 0.728 | 915.4 | 1.355 | 0.001662 | 815 |
| multitask | copy_on_write | 2 | 0.716 | 917.6 | 1.338 | 0.001462 | 915 |

Interpretation:

- Raw QAG was led by specialist routing with `standard_lora` and
  `copy_on_write`.
- Cache-footprint efficiency was led by `activated_lora`, because it preserved
  the lowest simulated memory footprint.
- The result supports the repo thesis, with an important nuance: the best
  strategy depends on whether the serving objective weights raw goodput or
  goodput per cache footprint more heavily.

## Caveats

- The task adapters were trained quickly on generated public-domain-style data;
  this is an engineering validation path, not a paper-grade dataset.
- Cache models are still simulators layered around real vLLM requests; the
  activated-LoRA and copy-on-write paths are not vLLM kernel implementations.
- The real server provides prefix-cache metrics, but adapter-specific cache
  namespaces are approximated by the benchmark cache model.
- Stronger claims need a larger source-backed held-out dataset, more models,
  and concurrent-load experiments.
