# gcloud vLLM runbook

Use gcloud when you need a real vLLM server with GPU memory, adapter loading,
and OpenAI-compatible `/v1/chat/completions`.

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

Example shape:

```bash
gcloud compute instances create adapter-cache-vllm \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu124 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB
```

GPU availability varies by region. If this fails, choose another zone or GPU
type that has quota in your project.

## 3. Start vLLM on the VM

SSH in:

```bash
gcloud compute ssh adapter-cache-vllm
```

Install or update vLLM in an environment appropriate for the image:

```bash
python -m pip install -U vllm
```

Start an OpenAI-compatible server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora
```

Adapter registration depends on the model and vLLM version. Put deployment
specific fields in `backend.extra_body` or expose adapters as server-side model
names.

## 4. Tunnel the API locally

From your laptop:

```bash
gcloud compute ssh adapter-cache-vllm -- -L 8000:localhost:8000
```

Then verify:

```bash
curl http://localhost:8000/v1/models
```

## 5. Run the benchmark

Edit `configs/benchmark/source_eval_vllm.yaml` for the served model and adapter
metadata, then run:

```bash
make vllm-source-eval
```

The run writes `requests.jsonl`, `summary.json`, `config_resolved.yaml`, and
`manifest.json` under `artifacts/runs/`.

## 6. Stop resources

Stop or delete the VM when finished:

```bash
gcloud compute instances stop adapter-cache-vllm
# or
gcloud compute instances delete adapter-cache-vllm
```
