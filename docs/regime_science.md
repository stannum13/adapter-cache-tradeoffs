# Regime Science V0

## Goal

Regime Science V0 turns the benchmark from pairwise policy comparison into a
small regime map:

> When does adapter specialization become worth its cache, latency, and capacity
> footprint?

This phase should remain CPU/mock-first. GPU runs come later, after the shape
generators, structure metrics, and regret tables are stable under unit tests and
dry-run budget gates.

## Regime Axes

The first regime suite varies request structure rather than model family:

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

- Do not claim broad model-quality results.
- Do not require GPU, vLLM, or internet for this phase.
- Do not add product CLI commands until the regime table shape is stable.
