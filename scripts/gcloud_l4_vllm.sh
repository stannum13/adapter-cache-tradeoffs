#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-adapter-cache-vllm-l4}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-150GB}"
SSH_KEY_FILE="${SSH_KEY_FILE:-.tmp-gcloud/adapter_cache_vllm_key}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LORA_BASE_MODEL="${LORA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LORA_REPO="${LORA_REPO:-uditjain/lori-qwen2.5-1.5b-medical}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  create           Create the L4 VM if it does not exist.
  start            Start an existing stopped VM.
  setup            Install NVIDIA driver, Docker, and NVIDIA container runtime.
  serve-base       Start vLLM with the base model only.
  serve-lora       Start vLLM with four registered LoRA module names.
  tunnel           Open localhost:${LOCAL_PORT} -> VM:${REMOTE_PORT}.
  run-base         Run source_eval_vllm + source_eval_vllm_l4_qwen overlays.
  run-lora         Run source_eval_vllm + source_eval_vllm_lora_qwen overlays.
  stop             Stop the VM.
  status           Print VM and vLLM status.

Required environment:
  PROJECT          GCP project id. Defaults to current gcloud project.

Common overrides:
  ZONE=${ZONE}
  INSTANCE=${INSTANCE}
  BASE_MODEL=${BASE_MODEL}
  LORA_BASE_MODEL=${LORA_BASE_MODEL}
  LORA_REPO=${LORA_REPO}
EOF
}

require_project() {
  if [[ -z "${PROJECT}" ]]; then
    echo "PROJECT is not set and no gcloud project is configured." >&2
    exit 2
  fi
}

ssh_vm() {
  gcloud compute ssh "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --ssh-key-file="${SSH_KEY_FILE}" \
    --command="$1"
}

create_vm() {
  require_project
  if gcloud compute instances describe "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" >/dev/null 2>&1; then
    echo "Instance ${INSTANCE} already exists."
    return
  fi
  gcloud compute instances create "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator=type=nvidia-l4,count=1 \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --provisioning-model=STANDARD \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type=pd-balanced \
    --scopes=https://www.googleapis.com/auth/cloud-platform
}

start_vm() {
  require_project
  gcloud compute instances start "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}"
}

setup_vm() {
  require_project
  ssh_vm '
    set -euo pipefail
    sudo apt-get update -y
    sudo apt-get install -y curl ca-certificates gnupg lsb-release ubuntu-drivers-common docker.io
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      sudo apt-get install -y nvidia-driver-570-server nvidia-utils-570-server || sudo ubuntu-drivers install --gpgpu
      echo "NVIDIA driver installed; rebooting. Re-run setup after SSH is available."
      sudo reboot
      exit 0
    fi
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl enable --now docker
    sudo systemctl restart docker
    nvidia-smi
    sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null
  '
}

serve_base() {
  require_project
  ssh_vm "
    set -euo pipefail
    sudo docker rm -f vllm-qwen >/dev/null 2>&1 || true
    sudo docker run -d --gpus all --name vllm-qwen --ipc=host \
      -p ${REMOTE_PORT}:8000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      ${VLLM_IMAGE} \
      --model ${LORA_BASE_MODEL} \
      --host 0.0.0.0 \
      --port 8000 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85
  "
}

serve_lora() {
  require_project
  ssh_vm "
    set -euo pipefail
    sudo docker rm -f vllm-qwen >/dev/null 2>&1 || true
    sudo docker run -d --gpus all --name vllm-qwen --ipc=host \
      -p ${REMOTE_PORT}:8000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      ${VLLM_IMAGE} \
      --model ${LORA_BASE_MODEL} \
      --host 0.0.0.0 \
      --port 8000 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85 \
      --enable-lora \
      --max-loras 4 \
      --max-lora-rank 64 \
      --lora-modules qa-lora=${LORA_REPO} json-lora=${LORA_REPO} summary-lora=${LORA_REPO} code-lora=${LORA_REPO}
  "
}

tunnel() {
  require_project
  gcloud compute ssh "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --ssh-key-file="${SSH_KEY_FILE}" \
    -- -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"
}

run_base() {
  uv run python -m adapter_cache_bench.bench.run_workload \
    --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_l4_qwen.yaml
}

run_lora() {
  uv run python -m adapter_cache_bench.bench.run_workload \
    --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_qwen.yaml
}

stop_vm() {
  require_project
  gcloud compute instances stop "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" --quiet
}

status_vm() {
  require_project
  gcloud compute instances list \
    --project="${PROJECT}" \
    --filter="name=${INSTANCE}" \
    --format="table(name,zone,status,machineType)"
  ssh_vm 'nvidia-smi || true; sudo docker ps -a --filter name=vllm-qwen --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || true'
}

case "${1:-}" in
  create) create_vm ;;
  start) start_vm ;;
  setup) setup_vm ;;
  serve-base) serve_base ;;
  serve-lora) serve_lora ;;
  tunnel) tunnel ;;
  run-base) run_base ;;
  run-lora) run_lora ;;
  stop) stop_vm ;;
  status) status_vm ;;
  ""|-h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
