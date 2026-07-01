# Automatic Build Plan

## Build Principle

Build the remaining work as a closed loop:

1. Define the claim or operational capability.
2. Add typed code and focused tests.
3. Produce or update machine-readable artifacts.
4. Regenerate reports/figures/tables if public surfaces changed.
5. Run `uv run ruff check .`, `uv run ruff format . --check`, and
   `uv run pytest tests -q`.
6. Commit one coherent change.
7. Record conclusions and residual risk before starting the next slice.

Use the detailed branch policy in
[`docs/autonomous_logic_tree.md`](autonomous_logic_tree.md) before each major
stage. That logic tree defines the four-critic review loop, evidence gates, and
fallback branches for mock evidence, real-server evidence, product tooling, and
claim bundles.

## Phase 1: Evidence Reliability

Goal: no GPU run should be scientifically useless just because it partially
failed.

Tasks:

- Add `status.json` lifecycle state for all benchmark runners.
- Write `config_resolved.yaml` and `manifest.json` at run start.
- Update status to `complete` or `failed` in `finally` blocks.
- Stream concurrent request rows as each request completes.
- Preserve error rows with request index, exception type, and message.
- Add manifest fields for backend model, serving URL, stream mode, adapter model
  names, concurrency, request spacing, and metric scope.

Exit criteria:

- Unit tests cover successful and failed sequential/concurrent runs.
- Failed runs keep enough artifacts to diagnose the failure.
- Existing benchmark tests still pass without GPU or internet.

## Phase 2: Evidence Bundles

Goal: public claims can be traced to sealed, hash-addressed evidence.

Tasks:

- Add an evidence bundle generator.
- Include selected run IDs, configs, summaries, manifests, status files, report
  paths, figure paths, and SHA256 hashes.
- Include git commit, dirty flag, generation timestamp, and excluded-file notes.
- Add a Make target for bundle generation.
- Document the bundle format.

Exit criteria:

- `bundle_manifest.json` can be generated from temporary test runs.
- Bundle generation does not require raw artifacts to be tracked in git.

## Phase 3: Claim Tables And Statistics

Goal: repeated-run evidence is reported as uncertainty, not only point
estimates.

Tasks:

- Add a claim evidence table generator.
- Add paired specialist-vs-multitask deltas where matched seeds/model aliases
  exist.
- Add 95% CI columns for repeated summaries.
- Add task-stratified quality reporting from request logs.
- Label p95 values as run-level or request-level depending on computation.

Exit criteria:

- CSV tests cover missing data, single-run data, repeated runs, and paired
  deltas.
- Public docs avoid paper-grade claims when only proxy quality is available.

## Phase 4: Resumable Sweeps And Overnight Safety

Goal: overnight experiments are stateful, resumable, and cost-aware.

Status: initial sweep state support is implemented for matrix, concurrency, and
exhaustive sweep runners. The runners now emit `sweep_plan.json`,
`sweep_status.json`, and `sweep_summary.md`; support `--resume`,
`--continue-on-error`, `--dry-run`, and budget gates; and can record cloud
provenance in child manifests through `ACB_CLOUD_*` environment variables.

Tasks:

- Add child-run checkpointing to sweep runners.
- Add `--resume` and `--continue-on-error` behavior.
- Validate markers by checking expected run artifacts, not marker files alone.
- Add budget preflight knobs for max wall clock, runs, requests, and expected
  GPU hours.
- Add cloud/runtime provenance hooks and TTL/shutdown documentation.

Exit criteria:

- Interrupted sweeps can resume missing children.
- Morning summaries distinguish complete, skipped, failed, and retried work.

## Phase 5: External User Path

Goal: a new user can run and understand the project quickly.

Status: the CPU-first quickstart, minimal `acb run`, `acb report`,
`acb bundle`, and non-invasive `acb doctor --config` preflight are implemented.
`recommend` remains deferred until real-server evidence and claim tables are
strong enough to support recommendations.

Tasks:

- Keep `docs/quickstart.md` current with CPU smoke, benchmark-v0, custom JSONL
  eval, vLLM, report generation, evidence bundle, and `acb doctor` paths.
- Keep `acb run`, `acb report`, `acb bundle`, and `acb doctor` as thin wrappers
  over stable underlying workflows.
- Add `recommend` only after real-server evidence and claim tables are strong
  enough to support recommendations.

Exit criteria:

- A new user can complete a CPU smoke path in under 20 minutes.
- The quickstart explains which tests require GPU, vLLM, or external model
  servers and keeps them optional.

## Phase 6: Regime Science

Goal: convert the benchmark into an explanatory regime map.

Tasks:

- Add workload-shape generators for uniform, Zipfian, bursty/session-local,
  phase-shifted, and adversarial adapter traffic.
- Report entropy, Gini, reuse distance, overlap, and policy regret.
- Add prefix-cache disabled and cold-cache baselines.
- Expand capacity frontier across context length, adapter rank/count, GPU memory,
  and memory utilization.
- Add training-budget parity ablations.

Exit criteria:

- A figure/table can state when cache reuse helps, when it fails, and how far
  each policy is from an oracle baseline.

## Current Parallel Workstreams

- **A. Lifecycle artifacts:** runner start/failure/complete state and enriched
  manifests.
- **B. Evidence bundles:** bundle manifest generator and tests.
- **C. Claim tables:** repeated-run confidence and paired-delta summaries.

Each workstream should land as an atomic commit with tests and a concise risk
note. After those commits, run the full repo checks and hold a second debate
council on the long-term direction.
