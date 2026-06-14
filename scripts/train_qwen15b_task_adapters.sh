#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SFT_DIR="${SFT_DIR:-artifacts/sft/public_domain_large}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/adapters}"
MAX_STEPS="${MAX_STEPS:-60}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
MAX_LENGTH="${MAX_LENGTH:-768}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

python3 -m pip install -q -e ".[real]"

for task in qa json summary code; do
  echo "TRAINING-${task}"
  python3 experimental/training/train_lora.py \
    --base-model "${BASE_MODEL}" \
    --train-file "${SFT_DIR}/train_${task}.jsonl" \
    --adapter-id "${task}" \
    --output-dir "${OUTPUT_DIR}/qwen15b-${task}" \
    --max-steps "${MAX_STEPS}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --max-length "${MAX_LENGTH}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
done
