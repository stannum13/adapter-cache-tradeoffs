# Trained adapter experiment

This path tests the quality side of the thesis with real LoRA adapters.

The clean comparison uses one base model for both conditions:

- Base: `Qwen/Qwen2.5-1.5B-Instruct`
- Specialist LoRAs: `qa-lora`, `json-lora`, `summary-lora`, `code-lora`
- Eval: `data/eval/source_eval.jsonl`

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

Serve the trained adapters with vLLM:

```bash
PROJECT=<project-id> \
LORA_MODULES="qa-lora=/home/shiva/adapters/qwen15b-qa json-lora=/home/shiva/adapters/qwen15b-json summary-lora=/home/shiva/adapters/qwen15b-summary code-lora=/home/shiva/adapters/qwen15b-code" \
./scripts/gcloud_l4_vllm.sh serve-lora
```

Run the same-base base-model and trained-adapter evals:

```bash
make vllm-source-eval-l4-qwen15b
make vllm-source-eval-lora-trained-qwen15b
make vllm-heldout-qwen15b
make vllm-heldout-lora-trained-qwen15b
```

The result supports the full hypothesis only if the trained specialist adapters
improve quality enough to offset their cache and latency footprint relative to
the same base model.
