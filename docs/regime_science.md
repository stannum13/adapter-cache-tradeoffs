# Regime Science V0

## Goal

Regime Science V0 turns the benchmark from pairwise policy comparison into a
small regime map:

> When does adapter specialization become worth its cache, latency, and capacity
> footprint?

This phase should remain CPU/mock-first. GPU runs come later, after the shape
generators, structure metrics, and regret tables are stable under unit tests and
dry-run budget gates.

## Branch A Claim Boundary

Branch A is a claim-bounded regime-science update. It supports simulator claims
about workload-dependent policy behavior; it does not support new real-serving
claims about vLLM, GPU memory, production throughput, or server-side
prefix-cache dynamics.

| supported simulator-backed claim | unsupported real-serving claim | evidence-required next step |
| --- | --- | --- |
| The deterministic CPU/mock suite can generate uniform, Zipfian, bursty, phase-shifted, and adversarial regimes with measurable structure differences. | These regimes cover production traffic distributions or customer workloads. | Add real request traces or public workload families, then compare their structure metrics to the synthetic regimes. |
| Under simulator cache controls, routing and cache policies have workload-dependent regret. | The same policy ordering will hold in vLLM or another model server. | Run a reset-isolated G8 bridge over the claim-critical regimes with repeated seeds where feasible. |
| `warm`, `cold`, and `prefix_disabled` separate benchmark-side cache mechanisms for the mock runner. | These controls reproduce server batching, prefix-cache keys, adapter-loading overhead, or GPU memory pressure. | Capture server reset settings, launch parameters, prefix/cache counters, latency, and memory metrics in real-server manifests. |
| Structure metrics can explain why a policy wins or fails in the simulator. | Structure metrics alone are sufficient for automated production recommendations. | Calibrate the simulator against real-serving measurements and keep recommendation logic deferred until the bridge passes. |

The publication boundary is therefore:

> Report the V0 result as a simulator-backed regime map for causal-transformer
> adapter-cache tradeoffs. Treat real-serving behavior as the next evidence
> step, not as an implication of the mock sweep.

## Regime Axes

The first regime suite varies request structure rather than causal-transformer
family:

- Uniform adapter/task mix.
- Zipfian adapter popularity.
- Bursty session-local traffic.
- Phase-shifted traffic where the dominant adapter changes over time.
- Adversarial churn that maximizes switching and reduces reuse.

Each generated workload should be deterministic from its seed and should encode
the intended structure through task order, session locality, and shared-prefix
reuse.

## Metrics

Each run should report workload-structure metrics alongside latency, cache, and
quality metrics:

- Adapter/task entropy.
- Gini concentration.
- Adapter switch rate.
- Mean reuse distance.
- Shared-prefix reuse ratio.
- Session locality.

These are explanatory variables. They should appear in summaries and downstream
tables so plots can explain why a policy won, not only that it won.

## Cache Conditions

The CPU/mock regime suite includes three benchmark-side cache conditions:

- `warm`: normal prefix-cache reuse across requests.
- `prefix_disabled`: no prefix-cache lookup or storage; cached-token and memory
  footprint metrics should remain zero.
- `cold`: no inter-request prefix-cache reuse, while each request still models
  the memory footprint it would populate during serving.

These conditions are simulator controls. Real vLLM or other model-server runs
still need server-side launch/reset settings that match the intended prefix-cache
behavior.

## Regret View

For each comparable regime group, report policy regret:

```text
regret = best_observed_qag - policy_qag
relative_regret = regret / best_observed_qag
```

When an oracle policy is present, use it as an explicit reference. Otherwise,
use the best observed policy in that regime and label the reference accordingly.

## V0 Deliverable

The first deliverable is a CPU/mock evidence bundle with:

1. `regime_v0` configs.
2. Dry-run sweep plan and budget summary.
3. Completed mock sweep.
4. Structure-metric table.
5. Policy-regret table.
6. One concise figure or table answering which policies fail under which
   workload structures.

Run the CPU/mock suite with:

```bash
make regime-v0-mock
make report
make evidence-bundle BUNDLE=regime-v0-mock RUN_GLOBS="regime_*" \
  REPORTS="reports/adapter-cache-tradeoffs.md" \
  FIGURES="reports/figures/regime_policy_failure_map.png"
```

The full V0 mock matrix covers 540 runs: 4 routers, 3 cache models, 3 cache
conditions, 5 workload regimes, and 3 seeds.

## Non-Goals

- Do not claim broad causal-transformer quality results.
- Do not claim real-serving behavior without a reset-isolated bridge bundle.
- Do not require GPU, vLLM, or internet for this phase.
- Do not add product CLI commands until the regime table shape is stable.
