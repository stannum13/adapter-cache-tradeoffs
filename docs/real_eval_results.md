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
```

## What ran

- 39 new vLLM-backed benchmark runs.
- 1,020 real model-server requests.
- 39 saved `backend_metrics_after.prom` snapshots.
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
