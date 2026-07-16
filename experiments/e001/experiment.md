# E001 Preregistration: Adapter Locality

**Status:** preregistered, not yet canonically run
**Date:** 2026-07-17
**Harness branch:** `refocus/upstream-e001-public`
**Upstream pins:** see `UPSTREAM.md`

## Question

When does adapter-aware routing improve quality-adjusted serving capacity once prefix locality, live load, adapter residency, and tail-latency constraints are measured in a real serving runtime?

## Hypothesis

A live-state adapter-aware router using measured prefix locality, adapter residency, active queue/in-flight state, and fixed quality priors will improve quality-adjusted SLO goodput over semantic-only, sticky-session, cache-static, multitask, and runtime-default baselines under TTFT constraints.

## Primary Substrate

Canonical evidence targets SGLang at the pinned commit in `UPSTREAM.md`, accessed through its OpenAI-compatible serving interface unless scheduler/cache internals require a focused fork. Vidur may be used before serving runs to rank candidate regimes. vLLM may be used only as a replication runtime for decisive comparisons.

## Baselines

- `base`: runtime/base-model route with no LoRA adapter model names.
- `semantic`: task-type specialist routing without cache/load scoring.
- `sticky_session`: session-affinity routing.
- `cache_static`: cache-aware routing with active-load weight fixed to zero.
- `multitask`: single multitask adapter route.
- `oracle`: upper-bound diagnostic only, excluded from promotion claims.

## Treatment

- `specialists`: cache-aware specialist routing with live active-load state enabled.
- Score weights are fixed by the config before canonical runs.
- Routing must happen at dispatch time, after concurrency admission, not as a full-workload preprocessing step.

## Fixed Variables

- Adapter IDs: `qa`, `json`, `summary`, `code`, plus `multitask` where the strategy requires it.
- Tenant/trust groups: two tenants or trust groups in the smoke plan, at least two in canonical runs.
- Prefix regimes: low, medium, and high shared-prefix overlap.
- Load regimes: at least two concurrency levels around saturation for canonical runs.
- Seeds: at least three repeated seeds for canonical serving runs.
- Prompt-token accounting: runtime prompt-token counts for server measurements where available; whitespace-token cache simulation remains labeled as simulator-side only.

## Primary Metrics

- p50/p95/p99 TTFT and end-to-end latency.
- Request and token throughput.
- SLO attainment.
- `quality_adjusted_slo_goodput`.
- Active in-flight/queue counts at routing decision time.
- Server cache-hit counters when the runtime exposes them.
- Adapter residency/load events when the runtime exposes them.
- GPU memory and adapter memory footprint when available.
- Task quality from exact or executable evaluators; weak lexical proxies cannot support headline claims.

## Required Controls

- Disable prefix caching while holding routing fixed.
- Shuffle or neutralize adapter-quality priors before promoting any quality-driven conclusion.
- Hold offered load fixed while varying prefix overlap.
- Hold prefix overlap fixed while varying active load.
- Repeat the decisive comparison in vLLM when feasible to test runtime specificity.

## Promotion Rule

Promote the result only if the treatment improves `quality_adjusted_slo_goodput` in more than one workload regime without a material task-quality regression or unexplained p99 latency regression. Otherwise publish the failure regimes and keep the README in honest current-status form.

## Falsification Rule

Falsify or demote the claim if the treatment only wins in simulator smoke, if improvement disappears when prefix caching is disabled, if gains depend on a quality proxy that is not task-valid, or if server cache/memory counters contradict the simulated cache explanation.

## Artifact Contract

Canonical results must be traceable through:

```text
configuration -> raw request records -> deterministic summary -> table/figure -> README statement
```

Required artifacts after canonical execution:

- `results/e001/manifest.json`
- `results/e001/summary.csv` or `results/e001/summary.json`
- `results/e001/figure.*`
- one routing score trace for a representative burst
- one latency-versus-load figure
- one prefix-overlap-versus-TTFT figure
- one table separating simulated and measured quantities
- a patch or branch reference for any SGLang fork changes

## Reproduction Commands

Fast local smoke:

```bash
./scripts/reproduce_e001.sh smoke
```

Canonical plan dry-run:

```bash
./scripts/reproduce_e001.sh canonical --dry-run
```

Canonical serving run, after starting the pinned SGLang server and any required adapters:

```bash
./scripts/reproduce_e001.sh canonical --resume --continue-on-error
```
