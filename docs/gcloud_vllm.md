# gcloud vLLM runbook

Use gcloud when you need a real vLLM server with GPU memory, adapter loading,
and OpenAI-compatible `/v1/chat/completions`. The helper script in
`scripts/gcloud_l4_vllm.sh` keeps the create/setup/serve/run/stop flow
repeatable.

For a first sanity check, prefer the local Hugging Face backend:

```bash
make transformers-source-eval
```

Use the GPU path when you want serving behavior closer to production.

## 1. Pick a project and zone

```bash
gcloud config set project <project-id>
gcloud config set compute/zone <zone>
```

Check quota and billing before creating a GPU VM.

## 2. Create a GPU VM

Scripted path:

```bash
PROJECT=<project-id> ZONE=us-central1-a ./scripts/gcloud_l4_vllm.sh create
PROJECT=<project-id> ZONE=us-central1-a ./scripts/gcloud_l4_vllm.sh setup
```

GPU availability varies by region. If this fails, choose another zone or GPU
type that has quota in your project.

For larger models, override the GPU and tensor-parallel settings instead of
using the default single-L4 shape. See
[large_model_benchmarking.md](large_model_benchmarking.md) for the staged
7B/14B/70B run path.

Spot/preemptible GPU quota and driver-ready images are also supported:

```bash
PROJECT=<project-id> \
ZONE=us-central1-c \
INSTANCE=adapter-cache-vllm-h100-spot \
MACHINE_TYPE=a3-highgpu-1g \
GPU_TYPE=nvidia-h100-80gb \
GPU_COUNT=1 \
PROVISIONING_MODEL=SPOT \
IMAGE_FAMILY=ubuntu-accelerator-2204-amd64-with-nvidia-580 \
IMAGE_PROJECT=ubuntu-os-accelerator-images \
./scripts/gcloud_l4_vllm.sh create
```

The accelerator image avoids a slow DKMS driver install, which matters for spot
VMs that can be reclaimed during setup.

## 3. Start vLLM on the VM

Base-model server:

```bash
PROJECT=<project-id> ./scripts/gcloud_l4_vllm.sh serve-base
```

Multi-LoRA server:

```bash
PROJECT=<project-id> ./scripts/gcloud_l4_vllm.sh serve-lora
```

The default LoRA smoke path serves `Qwen/Qwen2.5-1.5B-Instruct` and registers
the same public vLLM-compatible Qwen LoRA adapter under four model names:

```bash
qa-lora
json-lora
summary-lora
code-lora
```

Replace `LORA_REPO` or the script command if you have separately trained
specialist adapters. The benchmark config maps task adapters to vLLM model
names with `backend.adapter_model_names`.

## 4. Tunnel the API locally

From your laptop:

```bash
PROJECT=<project-id> ./scripts/gcloud_l4_vllm.sh tunnel
```

Then verify:

```bash
curl http://localhost:8000/v1/models
```

## 5. Run the benchmark

Export cloud provenance before launching GPU benchmarks. The runner copies
these values into `manifest.json` without calling gcloud directly:

```bash
export ACB_CLOUD_PROVIDER=gcp
export ACB_CLOUD_PROJECT=<project-id>
export ACB_CLOUD_ZONE=us-central1-a
export ACB_CLOUD_INSTANCE=adapter-cache-vllm-l4
export ACB_CLOUD_MACHINE_TYPE=g2-standard-8
export ACB_CLOUD_GPU_TYPE=nvidia-l4
export ACB_CLOUD_GPU_COUNT=1
export ACB_CLOUD_PROVISIONING_MODEL=STANDARD
export ACB_CLOUD_TTL_HOURS=8
export ACB_VLLM_IMAGE=vllm/vllm-openai:latest
```

Base-model run:

```bash
make vllm-source-eval-l4-qwen
```

LoRA-serving run:

```bash
make vllm-source-eval-lora-qwen
```

The run writes `requests.jsonl`, `summary.json`, `config_resolved.yaml`, and
`manifest.json` under `artifacts/runs/`. Current runners also write
`status.json`, and sweep runners write `sweep_plan.json`, `sweep_status.json`,
and `sweep_summary.md` under `artifacts/runs/_sweeps/<sweep-name>/`.

## 6. Stop resources

Stop or delete the VM when finished:

```bash
PROJECT=<project-id> ./scripts/gcloud_l4_vllm.sh stop
```

Stopping the VM stops GPU billing for the instance, but persistent disk storage
continues until the VM or disk is deleted.

For overnight work, set an explicit shutdown command in the surrounding shell or
loop. Prefer stop-on-exit for routine evidence runs and keep-alive-on-failure
only for debugging sessions:

```bash
trap 'PROJECT=<project-id> ./scripts/gcloud_l4_vllm.sh stop' EXIT
```

When launching VMs manually, label them with a TTL so external watchdogs can
find stale resources:

```bash
--labels=project=adapter-cache-bench,owner=<name>,purpose=benchmark,ttl_hours=8
```
