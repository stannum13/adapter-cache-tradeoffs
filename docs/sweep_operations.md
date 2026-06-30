# Sweep Operations

Use sweep state when an experiment expands one config into many child runs.
The sweep runners write parent state under:

```text
artifacts/runs/_sweeps/<sweep-name>/
```

Files:

- `sweep_plan.json`: deterministic child plan with child IDs, run names,
  dimensions, run directories, and planned request counts.
- `sweep_status.json`: live parent status with child states, attempts, errors,
  counts, budget information, and timestamps.
- `sweep_summary.md`: morning-readable summary for planned, complete, skipped,
  failed, and pending children.

## Supported Runners

The following entrypoints support sweep state:

```bash
uv run python -m adapter_cache_bench.bench.run_matrix
uv run python -m adapter_cache_bench.bench.run_concurrency_sweep
uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep
```

Each child uses its deterministic `run_name` as `run_id`, so resume behavior can
revisit the same child directory instead of creating timestamped duplicates.

## Dry Run

Use `--dry-run` before launching GPU work:

```bash
uv run python -m adapter_cache_bench.bench.run_matrix \
  --config configs/benchmark/small.yaml \
  --sweep-name small-dry-run \
  --dry-run
```

Dry-run writes the parent plan/status/summary but does not run child benchmarks
or generate reports.

## Budget Gates

Use budget gates to prevent accidental runaway matrices:

```bash
uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep \
  --config configs/benchmark/exhaustive_overlap_vllm_streaming.yaml \
  --sweep-name overlap-pilot \
  --dry-run \
  --max-runs 60 \
  --max-requests 5000 \
  --estimated-seconds-per-run 180 \
  --max-estimated-gpu-hours 3
```

If any limit is exceeded, the runner fails before launching children.

For the smallest reset-isolated real-server bridge, start with
`configs/benchmark/vllm_bridge_reset.yaml` and the runbook in
[vllm.md](vllm.md#run-the-minimal-g8-bridge). It plans 12 child runs and keeps
GPU/vLLM optional until the non-executing dry-run passes.

## Resume

Use `--resume` to skip children that already have all required artifacts and a
complete `status.json`:

```bash
uv run python -m adapter_cache_bench.bench.run_matrix \
  --config configs/benchmark/benchmark_v0_mock.yaml \
  --sweep-name benchmark-v0 \
  --resume
```

Skipped children are counted as `skipped` in `sweep_status.json`. Children with
missing artifacts, failed status, malformed status, or incomplete status are
rerun.

## Continue On Error

Use `--continue-on-error` for exploratory sweeps where one failed child should
not abort the whole matrix:

```bash
uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep \
  --config configs/benchmark/exhaustive_adapter_count_vllm_streaming.yaml \
  --sweep-name adapter-count-pilot \
  --continue-on-error
```

The parent status becomes `complete_with_failures` if any child fails. Failed
children include exception type and message in the parent status and summary.

## Evidence Discipline

Before a claim-supporting GPU run:

1. Run a dry-run with budget gates.
2. Confirm expected child count and request count in `sweep_summary.md`.
3. Export `ACB_CLOUD_*` provenance variables from
   [gcloud_vllm.md](gcloud_vllm.md).
4. Run with `--resume` and an explicit `--sweep-name`.
5. Build an evidence bundle after reports are generated.
