#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${mode}" in
  smoke)
    uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep \
      --config experiments/e001/configs/smoke.yaml \
      --sweep-name e001-smoke \
      --resume \
      "$@"
    ;;
  canonical)
    uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep \
      --config experiments/e001/configs/canonical.yaml \
      --sweep-name e001-canonical \
      --estimated-seconds-per-run 45 \
      --max-estimated-gpu-hours 5 \
      "$@"
    ;;
  *)
    echo "usage: $0 [smoke|canonical] [sweep options...]" >&2
    exit 2
    ;;
esac
