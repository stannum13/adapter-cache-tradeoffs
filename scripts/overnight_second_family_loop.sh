#!/usr/bin/env bash
set -euo pipefail

RUN_TS="$(date +%Y%m%d-%H%M%S)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OVERNIGHT_RUN_DIR="${OVERNIGHT_RUN_DIR:-artifacts/overnight/${RUN_TS}}"
COMMANDS_DIR="${OVERNIGHT_RUN_DIR}/commands"
MARKERS_DIR="${OVERNIGHT_RUN_DIR}/markers"
CONFIGS_DIR="${OVERNIGHT_RUN_DIR}/configs"
mkdir -p "${COMMANDS_DIR}" "${MARKERS_DIR}" "${CONFIGS_DIR}"

STATE_FILE="${OVERNIGHT_RUN_DIR}/state.json"
EVENTS_FILE="${OVERNIGHT_RUN_DIR}/events.jsonl"
SUMMARY_FILE="${OVERNIGHT_RUN_DIR}/summary.md"

DRY_RUN="${DRY_RUN:-0}"
AUTO_COMMIT="${AUTO_COMMIT:-0}"
MAX_RETRIES="${MAX_RETRIES:-1}"
INFRA_MAX_RETRIES="${INFRA_MAX_RETRIES:-2}"
RETRY_SLEEP_S="${RETRY_SLEEP_S:-30}"

WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-configs/benchmark/external_eval_vllm_template.yaml}"
SFT_DIR="${SFT_DIR:-${OVERNIGHT_RUN_DIR}/sft_external_eval}"
ADAPTERS_DIR="${ADAPTERS_DIR:-artifacts/adapters}"
mkdir -p "${ADAPTERS_DIR}"
ADAPTERS_DIR_ABS="$(cd "$(dirname "${ADAPTERS_DIR}")" && pwd)/$(basename "${ADAPTERS_DIR}")"

SECOND_BASE_MODEL="${SECOND_BASE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
SECOND_ALIAS="${SECOND_ALIAS:-tinyllama11b}"
SECOND_PREFIX="${SECOND_PREFIX:-tinyllama11b}"

QWEN_BASE_MODEL="${QWEN_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
QWEN_ALIAS="${QWEN_ALIAS:-qwen15b}"
QWEN_PREFIX="${QWEN_PREFIX:-qwen15b}"
TRAIN_QWEN_IF_MISSING="${TRAIN_QWEN_IF_MISSING:-1}"

LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_STEPS="${MAX_STEPS:-40}"
MULTITASK_MAX_STEPS="${MULTITASK_MAX_STEPS:-80}"
MAX_LENGTH="${MAX_LENGTH:-768}"
TRAIN_SEED="${TRAIN_SEED:-17}"

OVERNIGHT_REQUEST_COUNT="${OVERNIGHT_REQUEST_COUNT:-500}"
OVERNIGHT_MAX_CONCURRENCY="${OVERNIGHT_MAX_CONCURRENCY:-8}"
OVERNIGHT_SESSIONS="${OVERNIGHT_SESSIONS:-8}"
OVERNIGHT_TENANTS="${OVERNIGHT_TENANTS:-2}"
OVERNIGHT_SEEDS_CSV="${OVERNIGHT_SEEDS_CSV:-17,23,31}"
OVERNIGHT_STRATEGIES_CSV="${OVERNIGHT_STRATEGIES_CSV:-specialists,multitask}"

PORT="${PORT:-8000}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://localhost:${PORT}/health}"
VLLM_WARMUP_TIMEOUT_S="${VLLM_WARMUP_TIMEOUT_S:-600}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-second-family}"

ON_SUCCESS_COMMAND="${ON_SUCCESS_COMMAND:-}"
ON_FAILURE_COMMAND="${ON_FAILURE_COMMAND:-}"
ON_EXIT_COMMAND="${ON_EXIT_COMMAND:-}"

FINAL_STATUS="running"
CURRENT_PHASE="initializing"

quote() {
  printf "%q" "$1"
}

write_event() {
  local phase="$1"
  local status="$2"
  local attempt="$3"
  local duration_s="$4"
  local exit_code="$5"
  local log_path="$6"
  local command="$7"
  local detail="${8:-}"
  EVENT_PHASE="${phase}" EVENT_STATUS="${status}" EVENT_ATTEMPT="${attempt}" \
    EVENT_DURATION_S="${duration_s}" EVENT_EXIT_CODE="${exit_code}" \
    EVENT_LOG_PATH="${log_path}" EVENT_COMMAND="${command}" EVENT_DETAIL="${detail}" \
    python3 - <<'PY' >> "${EVENTS_FILE}"
import json
import os
from datetime import datetime, timezone

row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "phase": os.environ["EVENT_PHASE"],
    "status": os.environ["EVENT_STATUS"],
    "attempt": int(os.environ["EVENT_ATTEMPT"]),
    "duration_s": float(os.environ["EVENT_DURATION_S"]),
    "exit_code": int(os.environ["EVENT_EXIT_CODE"]),
    "log_path": os.environ["EVENT_LOG_PATH"],
    "command": os.environ["EVENT_COMMAND"],
}
detail = os.environ.get("EVENT_DETAIL", "")
if detail:
    row["detail"] = detail
print(json.dumps(row, sort_keys=True))
PY
}

write_state() {
  local phase="${1:-${CURRENT_PHASE}}"
  local status="${2:-${FINAL_STATUS}}"
  STATE_PHASE="${phase}" STATE_STATUS="${status}" STATE_DRY_RUN="${DRY_RUN}" \
    STATE_RUN_DIR="${OVERNIGHT_RUN_DIR}" STATE_WORKLOAD_CONFIG="${WORKLOAD_CONFIG}" \
    STATE_SECOND_BASE_MODEL="${SECOND_BASE_MODEL}" STATE_SECOND_ALIAS="${SECOND_ALIAS}" \
    STATE_QWEN_BASE_MODEL="${QWEN_BASE_MODEL}" STATE_QWEN_ALIAS="${QWEN_ALIAS}" \
    STATE_GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || true)" \
    python3 - <<'PY' > "${STATE_FILE}"
import json
import os
from datetime import datetime, timezone

state = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "status": os.environ["STATE_STATUS"],
    "current_phase": os.environ["STATE_PHASE"],
    "dry_run": os.environ["STATE_DRY_RUN"] == "1",
    "run_dir": os.environ["STATE_RUN_DIR"],
    "workload_config": os.environ["STATE_WORKLOAD_CONFIG"],
    "models": [
        {
            "role": "second_family",
            "base_model": os.environ["STATE_SECOND_BASE_MODEL"],
            "alias": os.environ["STATE_SECOND_ALIAS"],
        },
        {
            "role": "qwen_reference",
            "base_model": os.environ["STATE_QWEN_BASE_MODEL"],
            "alias": os.environ["STATE_QWEN_ALIAS"],
        },
    ],
    "git_head": os.environ["STATE_GIT_HEAD"],
}
print(json.dumps(state, indent=2, sort_keys=True))
PY
}

phase_attempts() {
  case "$1" in
    serve_*|wait_*|run_*|post_reports|post_readiness)
      echo "${INFRA_MAX_RETRIES}"
      ;;
    *)
      echo "${MAX_RETRIES}"
      ;;
  esac
}

run_phase() {
  local name="$1"
  local command="$2"
  local marker="${MARKERS_DIR}/${name}.done"
  local log_path="${COMMANDS_DIR}/${name}.log"
  local attempts
  attempts="$(phase_attempts "${name}")"
  CURRENT_PHASE="${name}"
  write_state "${name}" "running"

  if [[ -f "${marker}" ]]; then
    write_event "${name}" "skipped" "0" "0" "0" "${log_path}" "${command}" "marker exists"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf '$ %s\n' "${command}"
      printf 'DRY_RUN=1: command was not executed.\n'
    } > "${log_path}"
    touch "${marker}"
    write_event "${name}" "dry_run" "1" "0" "0" "${log_path}" "${command}"
    return 0
  fi

  local attempt exit_code start_ts end_ts duration
  for attempt in $(seq 1 "${attempts}"); do
    start_ts="$(date +%s)"
    printf '$ %s\n' "${command}" > "${log_path}"
    set +e
    (eval "${command}") >> "${log_path}" 2>&1
    exit_code=$?
    set -e
    end_ts="$(date +%s)"
    duration=$((end_ts - start_ts))
    if [[ "${exit_code}" -eq 0 ]]; then
      touch "${marker}"
      write_event "${name}" "ok" "${attempt}" "${duration}" "${exit_code}" "${log_path}" "${command}"
      return 0
    fi
    write_event "${name}" "failed_attempt" "${attempt}" "${duration}" "${exit_code}" "${log_path}" "${command}"
    if [[ "${attempt}" -lt "${attempts}" ]]; then
      sleep "${RETRY_SLEEP_S}"
    fi
  done

  FINAL_STATUS="failed"
  write_state "${name}" "failed"
  return "${exit_code}"
}

mark_phase_done() {
  local name="$1"
  local detail="$2"
  local marker="${MARKERS_DIR}/${name}.done"
  local log_path="${COMMANDS_DIR}/${name}.log"
  if [[ -f "${marker}" ]]; then
    write_event "${name}" "skipped" "0" "0" "0" "${log_path}" "no-op" "marker exists"
    return 0
  fi
  printf '%s\n' "${detail}" > "${log_path}"
  touch "${marker}"
  write_event "${name}" "ok" "0" "0" "0" "${log_path}" "no-op" "${detail}"
}

adapter_bundle_exists() {
  local prefix="$1"
  local task
  for task in qa json summary code multitask; do
    [[ -d "${ADAPTERS_DIR}/${prefix}-${task}" ]] || return 1
  done
}

require_adapter_bundle() {
  local prefix="$1"
  local missing=()
  local task
  for task in qa json summary code multitask; do
    if [[ ! -d "${ADAPTERS_DIR}/${prefix}-${task}" ]]; then
      missing+=("${ADAPTERS_DIR}/${prefix}-${task}")
    fi
  done
  if [[ "${#missing[@]}" -gt 0 ]]; then
    printf 'Missing adapter directories for prefix %s:\n' "${prefix}" >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi
}

lora_modules_for() {
  local prefix="$1"
  local alias="$2"
  printf '%s-qa-lora=/adapters/%s-qa ' "${alias}" "${prefix}"
  printf '%s-json-lora=/adapters/%s-json ' "${alias}" "${prefix}"
  printf '%s-summary-lora=/adapters/%s-summary ' "${alias}" "${prefix}"
  printf '%s-code-lora=/adapters/%s-code ' "${alias}" "${prefix}"
  printf '%s-multitask-lora=/adapters/%s-multitask' "${alias}" "${prefix}"
}

write_model_family_config() {
  local path="$1"
  local base_model="$2"
  local alias="$3"
  cat > "${path}" <<EOF
run_name: model-family-vllm-streaming
output_dir: artifacts/runs
workload:
  name: jsonl_eval
  dataset_path: data/eval/external_public_domain_eval.jsonl
  request_count: ${OVERNIGHT_REQUEST_COUNT}
  sessions: ${OVERNIGHT_SESSIONS}
  tenants: ${OVERNIGHT_TENANTS}
  max_tokens: 64
cache:
  model: activated_lora
  block_size: 8
router:
  policy: cache_aware
backend:
  kind: vllm
  base_url: http://localhost:${PORT}/v1
  api_key: EMPTY
  model: "${base_model}"
  temperature: 0.0
  stream: true
  ttft_slo_ms: 1000
  e2e_slo_ms: 12000
  scrape_metrics: true
  metrics_url: http://localhost:${PORT}/metrics
  max_concurrency: ${OVERNIGHT_MAX_CONCURRENCY}
  request_spacing_ms: 0
  extra_body: {}
matrix:
  strategies: [${OVERNIGHT_STRATEGIES_CSV}]
  concurrencies: [${OVERNIGHT_MAX_CONCURRENCY}]
  workloads: [jsonl_eval]
  caches: [activated_lora]
  adapter_counts: [4]
  tenants: [${OVERNIGHT_TENANTS}]
  isolation_scopes: [trust_group]
  seeds: [${OVERNIGHT_SEEDS_CSV}]
  models:
    - name: "${base_model}"
      alias: "${alias}"
      adapter_model_names:
        qa: "${alias}-qa-lora"
        json: "${alias}-json-lora"
        summary: "${alias}-summary-lora"
        code: "${alias}-code-lora"
        multitask: "${alias}-multitask-lora"
EOF
}

wait_for_vllm_health() {
  local deadline
  deadline=$((SECONDS + VLLM_WARMUP_TIMEOUT_S))
  until curl -fsS "${VLLM_HEALTH_URL}" >/dev/null; do
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      printf 'Timed out waiting for %s\n' "${VLLM_HEALTH_URL}" >&2
      return 1
    fi
    sleep 5
  done
}

write_env_snapshot() {
  {
    printf 'OVERNIGHT_RUN_DIR=%s\n' "${OVERNIGHT_RUN_DIR}"
    printf 'WORKLOAD_CONFIG=%s\n' "${WORKLOAD_CONFIG}"
    printf 'SECOND_BASE_MODEL=%s\n' "${SECOND_BASE_MODEL}"
    printf 'SECOND_ALIAS=%s\n' "${SECOND_ALIAS}"
    printf 'QWEN_BASE_MODEL=%s\n' "${QWEN_BASE_MODEL}"
    printf 'QWEN_ALIAS=%s\n' "${QWEN_ALIAS}"
    printf 'OVERNIGHT_REQUEST_COUNT=%s\n' "${OVERNIGHT_REQUEST_COUNT}"
    printf 'OVERNIGHT_SEEDS_CSV=%s\n' "${OVERNIGHT_SEEDS_CSV}"
    printf 'OVERNIGHT_STRATEGIES_CSV=%s\n' "${OVERNIGHT_STRATEGIES_CSV}"
    printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
  } > "${OVERNIGHT_RUN_DIR}/env.snapshot"
}

write_summary() {
  {
    printf '# Overnight second-family loop\n\n'
    printf '%s\n' "- status: \`${FINAL_STATUS}\`"
    printf '%s\n' "- run dir: \`${OVERNIGHT_RUN_DIR}\`"
    printf '%s\n' "- second family: \`${SECOND_BASE_MODEL}\` as \`${SECOND_ALIAS}\`"
    printf '%s\n' "- qwen reference: \`${QWEN_BASE_MODEL}\` as \`${QWEN_ALIAS}\`"
    printf '%s\n' "- request count per run: \`${OVERNIGHT_REQUEST_COUNT}\`"
    printf '%s\n\n' "- seeds: \`${OVERNIGHT_SEEDS_CSV}\`"
    printf '## Completed markers\n\n'
    find "${MARKERS_DIR}" -type f -name '*.done' -print | sort | sed 's#^.*/#- #; s#\.done$##'
    if [[ -f "${OVERNIGHT_RUN_DIR}/research_readiness.md" ]]; then
      printf '\n## Research readiness\n\n'
      cat "${OVERNIGHT_RUN_DIR}/research_readiness.md"
      printf '\n'
    fi
    printf '\n## Last events\n\n'
    tail -n 20 "${EVENTS_FILE}" 2>/dev/null || true
  } > "${SUMMARY_FILE}"
}

cleanup() {
  local exit_code=$?
  local state_phase="${CURRENT_PHASE}"
  if [[ "${FINAL_STATUS}" == "running" ]]; then
    if [[ "${exit_code}" -eq 0 ]]; then
      FINAL_STATUS="complete"
    else
      FINAL_STATUS="failed"
    fi
  fi
  if [[ "${FINAL_STATUS}" == "complete" ]]; then
    state_phase="complete"
  fi
  write_state "${state_phase}" "${FINAL_STATUS}" || true
  write_summary || true
  if [[ "${FINAL_STATUS}" == "complete" && -n "${ON_SUCCESS_COMMAND}" ]]; then
    (eval "${ON_SUCCESS_COMMAND}") > "${COMMANDS_DIR}/on_success.log" 2>&1 || true
  fi
  if [[ "${FINAL_STATUS}" != "complete" && -n "${ON_FAILURE_COMMAND}" ]]; then
    (eval "${ON_FAILURE_COMMAND}") > "${COMMANDS_DIR}/on_failure.log" 2>&1 || true
  fi
  if [[ -n "${ON_EXIT_COMMAND}" ]]; then
    (eval "${ON_EXIT_COMMAND}") > "${COMMANDS_DIR}/on_exit.log" 2>&1 || true
  fi
}
trap cleanup EXIT

write_env_snapshot
write_state "initializing" "running"

write_model_family_config "${CONFIGS_DIR}/second_model_family.yaml" "${SECOND_BASE_MODEL}" "${SECOND_ALIAS}"
write_model_family_config "${CONFIGS_DIR}/qwen_model_family.yaml" "${QWEN_BASE_MODEL}" "${QWEN_ALIAS}"

run_phase preflight_repo "git status --short && uv run python --version && make validate-external-eval && uv run ruff check . && uv run ruff format . --check"
run_phase preflight_gpu "command -v docker && docker info >/dev/null && command -v nvidia-smi && nvidia-smi"
run_phase build_sft_split "uv run python experimental/training/build_sft_data.py --workload-config $(quote "${WORKLOAD_CONFIG}") --output-dir $(quote "${SFT_DIR}") --eval-fraction 0.2 --seed $(quote "${TRAIN_SEED}")"

if adapter_bundle_exists "${SECOND_PREFIX}"; then
  mark_phase_done train_second_family "Adapter bundle ${SECOND_PREFIX} already exists."
else
  run_phase train_second_family "BASE_MODEL=$(quote "${SECOND_BASE_MODEL}") SFT_DIR=$(quote "${SFT_DIR}") OUTPUT_DIR=$(quote "${ADAPTERS_DIR}") OUTPUT_PREFIX=$(quote "${SECOND_PREFIX}") LOAD_IN_4BIT=$(quote "${LOAD_IN_4BIT}") MAX_STEPS=$(quote "${MAX_STEPS}") MULTITASK_MAX_STEPS=$(quote "${MULTITASK_MAX_STEPS}") MAX_LENGTH=$(quote "${MAX_LENGTH}") TRAIN_SEED=$(quote "${TRAIN_SEED}") ./scripts/train_qwen15b_task_adapters.sh"
fi
run_phase verify_second_adapters "require_adapter_bundle $(quote "${SECOND_PREFIX}")"

if adapter_bundle_exists "${QWEN_PREFIX}"; then
  mark_phase_done ensure_qwen_family "Adapter bundle ${QWEN_PREFIX} already exists."
elif [[ "${TRAIN_QWEN_IF_MISSING}" == "1" ]]; then
  run_phase ensure_qwen_family "BASE_MODEL=$(quote "${QWEN_BASE_MODEL}") SFT_DIR=$(quote "${SFT_DIR}") OUTPUT_DIR=$(quote "${ADAPTERS_DIR}") OUTPUT_PREFIX=$(quote "${QWEN_PREFIX}") LOAD_IN_4BIT=$(quote "${LOAD_IN_4BIT}") MAX_STEPS=$(quote "${MAX_STEPS}") MULTITASK_MAX_STEPS=$(quote "${MULTITASK_MAX_STEPS}") MAX_LENGTH=$(quote "${MAX_LENGTH}") TRAIN_SEED=$(quote "${TRAIN_SEED}") ./scripts/train_qwen15b_task_adapters.sh"
else
  run_phase ensure_qwen_family "require_adapter_bundle $(quote "${QWEN_PREFIX}")"
fi
run_phase verify_qwen_adapters "require_adapter_bundle $(quote "${QWEN_PREFIX}")"

SECOND_LORA_MODULES="$(lora_modules_for "${SECOND_PREFIX}" "${SECOND_ALIAS}")"
run_phase serve_second_family "MODEL=$(quote "${SECOND_BASE_MODEL}") ADAPTERS_DIR=$(quote "${ADAPTERS_DIR_ABS}") LORA_MODULES=$(quote "${SECOND_LORA_MODULES}") CONTAINER_NAME=$(quote "${CONTAINER_NAME}-${SECOND_ALIAS}") PORT=$(quote "${PORT}") ./scripts/restart_local_vllm_lora.sh"
run_phase wait_second_family "wait_for_vllm_health"
run_phase run_second_family "uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config $(quote "${CONFIGS_DIR}/second_model_family.yaml")"

QWEN_LORA_MODULES="$(lora_modules_for "${QWEN_PREFIX}" "${QWEN_ALIAS}")"
run_phase serve_qwen_family "MODEL=$(quote "${QWEN_BASE_MODEL}") ADAPTERS_DIR=$(quote "${ADAPTERS_DIR_ABS}") LORA_MODULES=$(quote "${QWEN_LORA_MODULES}") CONTAINER_NAME=$(quote "${CONTAINER_NAME}-${QWEN_ALIAS}") PORT=$(quote "${PORT}") ./scripts/restart_local_vllm_lora.sh"
run_phase wait_qwen_family "wait_for_vllm_health"
run_phase run_qwen_family "uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config $(quote "${CONFIGS_DIR}/qwen_model_family.yaml")"

run_phase post_reports "make report release-report whitepaper-figure large-model-figures adapter-metrics capacity-frontier"
run_phase post_readiness "uv run python -m adapter_cache_bench.analysis.research_readiness --runs-dir artifacts/runs --format markdown | tee $(quote "${OVERNIGHT_RUN_DIR}/research_readiness.md")"
run_phase final_ruff_check "uv run ruff check ."
run_phase final_ruff_format "uv run ruff format . --check"
run_phase final_pytest "uv run pytest tests -q"

if [[ "${AUTO_COMMIT}" == "1" ]]; then
  run_phase auto_commit "git add -u && git diff --cached --quiet || git commit -m 'Add overnight second-family evidence artifacts'"
fi

FINAL_STATUS="complete"
write_state "complete" "complete"
write_summary
