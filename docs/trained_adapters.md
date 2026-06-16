# Trained adapter experiment

This path tests the quality side of the thesis with real LoRA adapters.

The clean comparison uses one base model for both conditions:

- Base: `Qwen/Qwen2.5-1.5B-Instruct`
- Specialist LoRAs: `qa-lora`, `json-lora`, `summary-lora`, `code-lora`
- Eval: `data/eval/source_eval.jsonl`

The larger-model comparison uses:

- Base: `Qwen/Qwen2.5-7B-Instruct`
- Specialist LoRAs: `qa-lora`, `json-lora`, `summary-lora`, `code-lora`
- Multitask LoRA: `multitask-lora`
- Eval: `artifacts/sft/public_domain_xlarge/eval_requests.jsonl`

Build training data from the larger public-domain fixture:

```bash
uv run --extra real python experimental/training/build_sft_data.py \
  --workload-config configs/benchmark/public_domain_eval_large.yaml \
  --output-dir artifacts/sft/public_domain_large \
  --eval-fraction 0.2
```

This writes both SFT rows and benchmark-compatible request rows:

- `train.jsonl`, `train_*.jsonl`: prompt/completion rows for training.
- `eval_requests.jsonl`: held-out request records for benchmark evaluation.

Train one adapter per task:

```bash
for task in qa json summary code; do
  uv run --extra real python experimental/training/train_lora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --train-file artifacts/sft/public_domain_large/train_${task}.jsonl \
    --adapter-id ${task} \
    --output-dir artifacts/adapters/qwen15b-${task} \
    --max-steps 80
done
```

For the 7B run on a 24 GB L4, use 4-bit LoRA training:

```bash
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
SFT_DIR=artifacts/sft/public_domain_xlarge \
OUTPUT_DIR=/home/shiva/adapters \
OUTPUT_PREFIX=qwen7b \
LOAD_IN_4BIT=1 \
MAX_STEPS=40 \
MULTITASK_MAX_STEPS=80 \
MAX_LENGTH=768 \
./scripts/train_qwen15b_task_adapters.sh
```

Train alternate 7B seeds for repeated-adapter checks:

```bash
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
SFT_DIR=artifacts/sft/public_domain_xlarge \
OUTPUT_DIR=/home/shiva/adapters \
OUTPUT_PREFIX=qwen7b-seed23 \
TRAIN_SEED=23 \
LOAD_IN_4BIT=1 \
MAX_STEPS=40 \
MULTITASK_MAX_STEPS=80 \
MAX_LENGTH=768 \
./scripts/train_qwen15b_task_adapters.sh
```

```bash
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
SFT_DIR=artifacts/sft/public_domain_xlarge \
OUTPUT_DIR=/home/shiva/adapters \
OUTPUT_PREFIX=qwen7b-seed31 \
TRAIN_SEED=31 \
LOAD_IN_4BIT=1 \
MAX_STEPS=40 \
MULTITASK_MAX_STEPS=80 \
MAX_LENGTH=768 \
./scripts/train_qwen15b_task_adapters.sh
```

Serve the trained adapters with vLLM:

```bash
PROJECT=<project-id> \
LORA_MODULES="qa-lora=/home/shiva/adapters/qwen15b-qa json-lora=/home/shiva/adapters/qwen15b-json summary-lora=/home/shiva/adapters/qwen15b-summary code-lora=/home/shiva/adapters/qwen15b-code" \
./scripts/gcloud_l4_vllm.sh serve-lora
```

For the trained 7B adapters:

```bash
PROJECT=<project-id> \
ZONE=<zone> \
INSTANCE=<instance> \
LORA_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.90 \
MAX_LORAS=5 \
LORA_MODULES="qa-lora=/home/shiva/adapters/qwen7b-qa json-lora=/home/shiva/adapters/qwen7b-json summary-lora=/home/shiva/adapters/qwen7b-summary code-lora=/home/shiva/adapters/qwen7b-code multitask-lora=/home/shiva/adapters/qwen7b-multitask" \
./scripts/gcloud_l4_vllm.sh serve-lora
```

Run the same-base base-model and trained-adapter evals:

```bash
make vllm-source-eval-l4-qwen15b
make vllm-source-eval-lora-trained-qwen15b
make vllm-heldout-qwen15b
make vllm-heldout-lora-trained-qwen15b
make vllm-heldout-lora-multitask-qwen15b
make vllm-heldout-trained-matrix-qwen15b
make vllm-heldout-trained-repeated-qwen15b
make vllm-heldout-xlarge-qwen15b-concurrent
make vllm-heldout-xlarge-lora-trained-qwen15b-concurrent
make vllm-heldout-xlarge-lora-multitask-qwen15b-concurrent
```

Run the 7B held-out comparison:

```bash
make vllm-heldout-xlarge-qwen7b
make vllm-heldout-xlarge-lora-trained-qwen7b
make vllm-heldout-xlarge-lora-multitask-qwen7b
make vllm-source-eval-l4-qwen7b
make vllm-source-eval-lora-trained-qwen7b
make vllm-source-eval-lora-multitask-qwen7b
make vllm-source-eval-lora-trained-qwen7b-seed23
make vllm-source-eval-lora-multitask-qwen7b-seed23
make vllm-source-eval-lora-trained-qwen7b-seed31
make vllm-source-eval-lora-multitask-qwen7b-seed31
make vllm-source-eval-expanded-qwen7b
make vllm-source-eval-expanded-lora-trained-qwen7b
make vllm-source-eval-expanded-lora-multitask-qwen7b
```

On one L4 at `max_model_len=4096`, serve one five-adapter seed group at a time.
The measured vLLM capacity probe started successfully with five LoRAs, but
failed with eight and ten registered LoRAs because there was not enough KV-cache
memory left for one max-length request.

The result supports the full hypothesis only if the trained specialist adapters
improve quality enough to offset their cache and latency footprint relative to
the same base model.

See [real_eval_results.md](real_eval_results.md) for one real vLLM snapshot.
