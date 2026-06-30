# Autonomous Logic Tree

This document defines the closed-loop build policy for autonomous work in this
repo. It is meant to be used before each overnight run, major code slice, report
update, or evidence bundle. The goal is to keep the system moving while making
claim scope, verification state, and failure handling explicit.

## Current State

The current branch has strong CPU/mock evidence for regime-science tooling:

- deterministic regime workload generators;
- workload structure metrics;
- policy regret analysis;
- warm, cold, and prefix-disabled cache conditions;
- a publication-grade regime failure-map figure;
- a 540-run local mock sweep and evidence bundle.

This supports claims about the benchmark harness, simulator behavior, and the
existence of workload-dependent policy regimes. It does not yet support strong
real-serving claims about vLLM, GPU memory behavior, production throughput, or
server-side prefix-cache dynamics without a fresh real-server evidence bundle.

## Four-Critic Loop

Launch four orthogonal actor-critics before every major stage. Each critic must
produce: `go`, `go_with_conditions`, or `stop`; the top two risks; the minimum
next test; and the strongest claim that would remain valid if the stage fails.

1. Scientific validity critic
   - Checks construct validity, falsifiability, baseline completeness, and claim
     language.
   - Blocks if a planned claim depends on evidence that is absent or simulated.

2. Systems reproducibility critic
   - Checks config source, budget preflight, resumability, artifact integrity,
     cloud provenance, and optional GPU boundaries.
   - Blocks GPU/cloud work if dry-run, budget, or resume behavior is unclear.

3. Product and user-path critic
   - Checks whether the next slice helps users run, inspect, reproduce, or trust
     the project.
   - Blocks polished product work when the underlying evidence state is not yet
     stable enough to expose.

4. Adversarial reviewer
   - Reads the work as a skeptical program committee reviewer.
   - Blocks if the report could be interpreted as stronger than the evidence,
     if a negative control is missing, or if failure modes are hidden.

Consensus rule:

- If all four say `go`, execute the smallest complete slice.
- If any say `go_with_conditions`, convert the condition into an explicit gate
  and verify it before claiming success.
- If any say `stop`, either narrow the claim until the stop is removed or switch
  to a lower-risk branch.
- If critics disagree, pick the branch that preserves the most falsifiability and
  requires the least irreversible cost.

## Root Classifier

Classify every next action into exactly one work type before starting:

- `code_build`: changes to source, tests, configs, Make targets, or CLI.
- `cpu_evidence`: local mock or CPU evidence generation.
- `gpu_evidence`: vLLM, CUDA, cloud, or external model-server evidence.
- `claim_bundle`: reports, figures, tables, evidence bundles, and PR language.
- `product_tooling`: quickstart, CLI wrappers, doctor checks, examples, or UX.

If ambiguous, default to `cpu_evidence` or `claim_bundle`. Do not default to
`gpu_evidence`.

## Global Loop

For each autonomous stage:

1. Classify the work type.
2. Launch the four critics and collect their branch recommendations.
3. Select one branch and write the branch hypothesis in one sentence.
4. For evidence work, run dry-run and budget checks before the real run.
5. Implement or execute the smallest complete slice.
6. Verify with the required command set for that work type.
7. Regenerate reports, figures, tables, and bundles if public surfaces changed.
8. Commit one coherent change, or leave generated ignored artifacts uncommitted
   with a manifest path in the status note.
9. Record residual risk and the strongest supported claim.
10. Re-enter the classifier and choose the next branch.

## Stage Gates

### G0: Claim Scope

Question: what claim would this stage make true?

Pass:

- The claim names the evidence source: mock simulator, CPU runner, vLLM server,
  external model API, or report-only synthesis.
- Unsupported claims are listed separately from supported claims.
- Public docs use causal-transformer terminology except where historical
  context requires otherwise.

Fail actions:

- Narrow the claim.
- Add a claim table with `supported`, `not_supported_yet`, and
  `evidence_required` columns.
- Do not run more expensive experiments until the claim boundary is clear.

### G1: Evidence Integrity

Question: can every public number be traced to sealed artifacts?

Pass:

- Each benchmark run has JSONL request logs, `summary.json`,
  `config_resolved.yaml`, and `manifest.json`.
- Sweeps have `sweep_plan.json`, `sweep_status.json`, and `sweep_summary.md`.
- Evidence bundles include hashes, git commit, dirty flag, included paths, and
  excluded-file notes.

Fail actions:

- Fix runner lifecycle or bundling before interpreting results.
- Resume only from artifact completeness, not marker files alone.
- Treat partial or failed runs as diagnostic evidence only.

### G2: Workload Construct Validity

Question: does the workload actually exercise the intended regime?

Pass:

- Workload summaries include entropy, Gini, reuse distance, overlap, adapter
  churn, and phase/session structure where relevant.
- Uniform, Zipfian, bursty, phase-shifted, and adversarial workloads separate in
  the reported structure metrics.
- The workload generator is deterministic under a seed.

Fail actions:

- Fix the generator or metrics before running larger sweeps.
- Add a negative-control workload.
- Do not treat policy differences as regime effects until the workload shape is
  validated.

### G3: Cache-Mechanism Validity

Question: does each cache condition isolate the mechanism it claims to isolate?

Pass:

- `warm` permits reuse.
- `cold` prevents inter-request reuse while still modeling current-request
  footprint.
- `prefix_disabled` disables lookup and store, and reports zero cached token
  ratio and memory footprint.
- Copy-on-write and related policies reset per-condition state correctly.

Fail actions:

- Fix the cache model and rerun the mock sweep.
- Add unit tests for any newly observed policy or condition interaction.
- Invalidate old evidence bundles that relied on the broken mechanism.

### G4: Experimental Design

Question: is the matrix large enough to answer the branch hypothesis and small
enough to complete safely?

Pass:

- Config is checked into `configs/benchmark/`.
- Dry-run reports expected runs, requests, wall time, and GPU hours.
- The selected factors map directly to the branch hypothesis.
- Repeated seeds exist for claim-critical comparisons, or the claim explicitly
  says it is exploratory.

Fail actions:

- Reduce to a bridge subset before scaling out.
- Add seeds only to the claim-critical slice.
- Split exploratory and confirmatory sweeps into separate bundles.

### G5: Outcome Validity

Question: do measured outcomes match the claim?

Pass:

- Throughput, TTFT, latency, memory, quality, and cache metrics are labeled by
  source and scope.
- p95 values state whether they are run-level or request-level.
- Proxy quality is not described as task-grade answer quality.
- Real-server results include server-side counters or clearly state that they do
  not.

Fail actions:

- Downgrade the claim.
- Add request-level or task-stratified summaries.
- Instrument missing counters before making serving claims.

### G6: Falsification And Negative Controls

Question: what result would make the current explanation wrong?

Pass:

- At least one negative control is present for claim-critical evidence.
- Prefix-disabled and cold-cache conditions are included when claiming prefix
  reuse effects.
- A baseline that should not benefit from the mechanism is visible in the table
  or figure.

Fail actions:

- Add the missing negative control.
- Move the claim from conclusion to hypothesis.
- Show failure cases in the main report, not only in an appendix.

### G7: Statistical Robustness

Question: are differences stable enough to report?

Pass:

- Repeated runs exist for the claim-critical comparisons.
- Reports include uncertainty intervals or label single-run results as such.
- Paired deltas are used where matched seeds and configurations exist.
- Policy regret is reported against a clear oracle or baseline source.

Fail actions:

- Add seeds to the smallest matrix that tests the claim.
- Use qualitative language for single-run evidence.
- Avoid ranking policies when confidence intervals overlap materially.

### G8: Real-Server Bridge

Question: does simulated behavior survive contact with vLLM or another real
server?

Pass:

- A reset-isolated real-server subset exists for the claim-critical workloads.
- Conditions are comparable to mock conditions or the gap is explicitly named.
- Manifests include backend model, serving URL, adapter names, stream mode,
  runtime provenance, and cloud metadata when applicable.
- Server-side prefix/cache counters are captured when making cache claims.

Fail actions:

- Run a minimal bridge subset: two or three workloads, two or three policies,
  two cache-relevant conditions, and repeated seeds if feasible.
- If counters are unavailable, downgrade to client-observed serving behavior.
- If real-server results contradict mock results, calibrate or narrow the mock
  claim instead of averaging the disagreement away.

### G9: External Validity

Question: does the finding generalize beyond the current toy or bridge setup?

Pass:

- The evidence spans multiple context lengths, adapter counts, adapter ranks,
  request rates, or memory pressure regimes as required by the claim.
- Real adapters, trained adapters, or public datasets are used when the claim
  refers to real workloads.
- Limits and non-results are stated near the headline claim.

Fail actions:

- Keep the result framed as a benchmark method or internal regime map.
- Add one external workload family at a time.
- Do not widen the claim until the new family passes G1 through G8.

### G10: Product Readiness

Question: will a new user be able to reproduce and interpret the work?

Pass:

- A quickstart covers CPU smoke, benchmark config, custom JSONL eval, vLLM, and
  report generation.
- Optional GPU, vLLM, and external model-server paths are clearly optional.
- CLI wrappers expose stable workflows only after the underlying lifecycle is
  reliable.
- `doctor` checks are non-invasive and explain missing optional dependencies.

Fail actions:

- Add docs or `doctor` checks before adding a recommendation engine.
- Keep `recommend` deferred until real evidence and claim tables are strong.
- Prefer reproducible configs over ad hoc commands.

## Scenario Branches

### Branch A: Mock Regime Evidence Passes, Real Bridge Missing

State:

- G0 through G7 pass for simulator-backed evidence.
- G8 is missing or explicitly not attempted.

Action:

- Close or merge only mock/regime-science claims.
- Add a supported-claim table that says real-serving claims require G8.
- Next best slice is a minimal vLLM bridge subset if GPU access is available.
- If GPU access is unavailable, improve quickstart, claim tables, and bundle
  readability.

### Branch B: Workload Metrics Do Not Separate Regimes

State:

- Runs complete, but G2 fails.

Action:

- Stop interpreting policy deltas.
- Fix workload generators and structure metrics.
- Add tests that assert expected ordering for entropy, Gini, reuse distance, and
  adapter churn.
- Rerun the CPU/mock matrix after the fix.

### Branch C: Cache Controls Contradict Their Definitions

State:

- Prefix-disabled shows cached-token or memory reuse, cold cache leaks
  inter-request state, or warm cache fails to reuse.

Action:

- Fix `cache_models.py` or runner condition plumbing.
- Add unit tests for each failed policy-condition pair.
- Rebuild the affected evidence bundle.
- Mark prior affected reports stale.

### Branch D: Dry-Run Or Budget Fails

State:

- Planned run exceeds wall time, request count, estimated GPU hours, or cloud
  safety limits.

Action:

- Do not start the run.
- Reduce the matrix to the smallest branch-hypothesis subset.
- Prefer fewer policies and more seeds for confirmatory claims.
- Require TTL or shutdown documentation before cloud GPU runs.

### Branch E: Partial Sweep Failure

State:

- Some child runs fail or artifacts are incomplete.

Action:

- Resume missing children if artifacts prove they are incomplete.
- Triage failures by backend, workload, policy, and cache condition.
- Treat the sweep as diagnostic until rerun or explicitly scoped.
- Preserve failed child manifests and request rows for debugging.

### Branch F: Server Counters Unavailable

State:

- Real-server run completes, but server-side cache or memory counters are absent.

Action:

- Downgrade from mechanism claim to client-observed behavior.
- Add instrumentation before claiming prefix-cache causality.
- Keep the evidence bundle, but label the metric gap in report tables.

### Branch G: Real Server Contradicts Simulator

State:

- G8 produces a directionally different result from the mock matrix.

Action:

- Treat this as the highest-value result, not a failure.
- Add a simulator-calibration issue or phase.
- Identify which mechanism differs: scheduling, batching, prefix cache,
  adapter-loading overhead, memory pressure, or quality path.
- Narrow public claims to the shared subset that survives both systems.

### Branch H: Real Bridge Passes

State:

- G8 passes for a reset-isolated subset.

Action:

- Scale one axis at a time: capacity, concurrency, context length, adapter rank,
  adapter count, or external workload.
- Add repeated seeds around the strongest and weakest regimes.
- Promote report language from simulator-backed hypothesis to scoped serving
  evidence only for covered conditions.

### Branch I: User Reproducibility Blocks Trust

State:

- Evidence exists, but a new user cannot rerun or inspect it quickly.

Action:

- Build `docs/quickstart.md`.
- Add minimal `acb run --config`, `acb report`, and `acb bundle` wrappers only
  around stable commands.
- Add `acb doctor --config` for dependency and config preflight.
- Defer `acb recommend` until G8 and G10 both pass for enough data.

## Work-Type Verification

Use the strictest applicable verification set.

For `code_build`:

- `uv run pytest tests -q`
- `uv run ruff check .`
- `uv run ruff format . --check`

For `cpu_evidence`:

- dry-run or budget preflight when available;
- run the selected config;
- inspect `sweep_status.json` or per-run artifacts;
- regenerate reports if public surfaces changed;
- generate or update an evidence bundle.

For `gpu_evidence`:

- all `cpu_evidence` checks;
- explicit GPU/cloud budget limits;
- runtime provenance in manifests;
- optional dependency failures must not break unit tests;
- cloud TTL or shutdown plan before overnight runs.

For `claim_bundle`:

- regenerate reports, tables, figures, and bundle manifests;
- verify hashes and included paths;
- inspect figure readability if visual output changed;
- run `git diff --check` for docs-only changes.

For `product_tooling`:

- unit tests for CLI/config behavior;
- CPU smoke path documented;
- optional GPU/vLLM paths remain optional;
- full repo checks before finalizing.

## Immediate Recommendation

The current best branch is Branch A.

Short-term action:

1. Freeze PR #2 language to simulator-backed regime-science claims.
2. Add or update a visible claim table that separates supported, unsupported,
   and evidence-required claims.
3. If GPU access is available, run a reset-isolated minimal vLLM bridge subset:
   two or three workloads, two or three policies, warm plus one cache-control
   condition, and repeated seeds where feasible.
4. If GPU access is not available, improve the external user path with
   quickstart and minimal stable wrappers before adding recommendation logic.

Long-term direction:

1. Use the vLLM bridge to decide whether the simulator is explanatory,
   predictive, or only diagnostic.
2. Scale one real-serving axis at a time after G8 passes.
3. Make negative controls and uncertainty first-class report outputs.
4. Defer automated policy recommendation until real-server evidence, claim
   tables, and user-facing reproduction paths are all stable.
