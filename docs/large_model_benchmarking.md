# Large-model benchmarking

Large models are useful for this project, but they answer two different
questions.

## Question A: does cache locality matter more as the model gets larger?

This can be measured before training any new adapters. Serve a larger causal
transformer through vLLM, run the same shared-prefix workloads, and compare
TTFT, throughput, server prefix-cache hit rate, and benchmark cache accounting.

Run path:

```bash
make vllm-large-model
```

The template is [large_model_vllm_template.yaml](../configs/benchmark/large_model_vllm_template.yaml).
It is base-only by default. That is intentional: it isolates the systems effect
of prefill cost, shared-prefix reuse, concurrency, and tenant isolation without
waiting for large-model adapter training.

For the first real 7B pilot result, see
[large_model_results.md](large_model_results.md).

Recommended progression:

| tier | purpose | expected hardware shape |
| --- | --- | --- |
| 7B/8B | first cross-size serving run | one high-memory GPU or quantized single-GPU setup |
| 14B/32B | stronger cache/SLO scaling signal | larger single GPU or two-way tensor parallel |
| 70B-class | serious serving stress test | multi-GPU tensor parallel or quantized serving |

Do not compare quality across model sizes unless the evaluation data, decoding
settings, and scoring path are held fixed.

## Question B: does adapter specialization still win on a larger model?

This requires matching specialist and multitask adapters for the larger base
model. Use [model_family_vllm_template.yaml](../configs/benchmark/model_family_vllm_template.yaml)
after those adapters exist.

Required evidence:

- Same held-out dataset for all model families.
- Same SFT protocol for specialist and multitask adapters.
- Same vLLM serving mode, streaming TTFT, and SLOs.
- Adapter names mapped through `backend.adapter_model_names`.
- Server metrics scraped, preferably with `configs/benchmark/local_vllm_reset.yaml`
  when collecting per-condition cache evidence.

## Serving large models on GCP

The helper script accepts GPU and tensor-parallel overrides:

```bash
PROJECT=<project-id> \
INSTANCE=adapter-cache-vllm-large \
MACHINE_TYPE=<gpu-machine-type> \
GPU_TYPE=<gpu-type> \
GPU_COUNT=<count> \
BASE_MODEL=<model-name> \
MAX_MODEL_LEN=4096 \
TENSOR_PARALLEL_SIZE=<count> \
./scripts/gcloud_l4_vllm.sh create
```

Then:

```bash
PROJECT=<project-id> INSTANCE=adapter-cache-vllm-large ./scripts/gcloud_l4_vllm.sh setup
PROJECT=<project-id> INSTANCE=adapter-cache-vllm-large ./scripts/gcloud_l4_vllm.sh serve-base
PROJECT=<project-id> INSTANCE=adapter-cache-vllm-large ./scripts/gcloud_l4_vllm.sh tunnel
make vllm-large-model
```

The exact GPU type and machine type depend on quota and regional availability.
Do not leave these instances running after a sweep:

```bash
PROJECT=<project-id> INSTANCE=adapter-cache-vllm-large ./scripts/gcloud_l4_vllm.sh stop
```

## Interpretation

For large base-only runs, the claim is:

> Larger causal transformers make prefill more expensive, so shared-prefix cache
> locality should become more valuable under the same workload shape.

For large adapter runs, the stronger claim is:

> Specialist adapters are worth their cache footprint only when their quality
> gain beats the extra namespace fragmentation, latency, and memory cost.

The second claim is the paper-grade result. The first claim is the cheaper
systems scaling run that tells us whether spending on large adapter training is
worth it.
