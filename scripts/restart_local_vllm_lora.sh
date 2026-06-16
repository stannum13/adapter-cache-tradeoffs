#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
LORA_MODULES="${LORA_MODULES:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
ADAPTERS_DIR="${ADAPTERS_DIR:-}"

if [[ -z "${LORA_MODULES}" ]]; then
  cat >&2 <<'EOF'
LORA_MODULES is required.

Example:
LORA_MODULES="qa-lora=/adapters/qwen15b-qa json-lora=/adapters/qwen15b-json summary-lora=/adapters/qwen15b-summary code-lora=/adapters/qwen15b-code multitask-lora=/adapters/qwen15b-multitask" \
ADAPTERS_DIR="$HOME/adapter-cache-bench-train/artifacts/adapters" \
./scripts/restart_local_vllm_lora.sh
EOF
  exit 2
fi

mount_args=(-v "${HF_CACHE_DIR}:/root/.cache/huggingface")
if [[ -n "${ADAPTERS_DIR}" ]]; then
  mount_args+=(-v "${ADAPTERS_DIR}:/adapters")
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d --gpus all --name "${CONTAINER_NAME}" --ipc=host \
  -p "${PORT}:8000" \
  "${mount_args[@]}" \
  "${VLLM_IMAGE}" \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-lora \
  --max-loras 5 \
  --max-lora-rank 64 \
  --lora-modules ${LORA_MODULES}
