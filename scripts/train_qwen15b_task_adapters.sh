#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SFT_DIR="${SFT_DIR:-artifacts/sft/public_domain_large}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/adapters}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen15b}"
MAX_STEPS="${MAX_STEPS:-60}"
MULTITASK_MAX_STEPS="${MULTITASK_MAX_STEPS:-${MAX_STEPS}}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
MAX_LENGTH="${MAX_LENGTH:-768}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
TRAIN_MULTITASK="${TRAIN_MULTITASK:-1}"

extra_args=()
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  extra_args+=(--load-in-4bit)
fi

python3 -m pip install -q -e ".[real]"
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  python3 -m pip install -q bitsandbytes
fi

for task in qa json summary code; do
  echo "TRAINING-${task}"
  python3 experimental/training/train_lora.py \
    --base-model "${BASE_MODEL}" \
    --train-file "${SFT_DIR}/train_${task}.jsonl" \
    --adapter-id "${task}" \
    --output-dir "${OUTPUT_DIR}/${OUTPUT_PREFIX}-${task}" \
    --max-steps "${MAX_STEPS}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --max-length "${MAX_LENGTH}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    "${extra_args[@]}"
done

if [[ "${TRAIN_MULTITASK}" == "1" ]]; then
  echo "TRAINING-multitask"
  python3 -m experimental.training.train_multitask_lora \
    --base-model "${BASE_MODEL}" \
    --train-file "${SFT_DIR}/train.jsonl" \
    --output-dir "${OUTPUT_DIR}/${OUTPUT_PREFIX}-multitask" \
    --max-steps "${MULTITASK_MAX_STEPS}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --max-length "${MAX_LENGTH}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    "${extra_args[@]}"
fi
