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

Base-model run:

```bash
make vllm-source-eval-l4-qwen
```

LoRA-serving run:

```bash
make vllm-source-eval-lora-qwen
```

The run writes `requests.jsonl`, `summary.json`, `config_resolved.yaml`, and
`manifest.json` under `artifacts/runs/`.

## 6. Stop resources

Stop or delete the VM when finished:

```bash
PROJECT=<project-id> ./scripts/gcloud_l4_vllm.sh stop
```

Stopping the VM stops GPU billing for the instance, but persistent disk storage
continues until the VM or disk is deleted.
