# Long-Term Concept: Adapter-Serving Frontier Lab

## Thesis

`adapter-cache-tradeoffs` should become a measurement and decision system for
causal-transformer adapter serving. Its long-term contribution is not that one
router or adapter strategy always wins. The contribution is a compact
explanation of when specialization, cache locality, SLO pressure, and adapter
capacity interact.

The long-term question is:

> Given traffic, adapters, hardware, and SLOs, which serving strategy should be
> used, and why?

## Research Direction

The project should evolve from isolated benchmark comparisons into a regime map.
Key latent variables should become first-class experiment dimensions:

- Adapter popularity distribution.
- Request burstiness and session locality.
- Adapter-switching entropy and reuse distance.
- Shared-prefix overlap.
- Context length and prefill/decode ratio.
- Adapter count, rank, and registration capacity.
- GPU memory and KV-cache pressure.
- Tenant isolation and trust-group constraints.

The scientific target is a small set of predictive rules:

- Cache reuse helps when workload structure is compressible.
- Specialization is worth serving only when quality gain exceeds SLO loss,
  fragmentation, and capacity cost.
- Multitask/base strategies win when latency or memory pressure dominates.
- Router choice matters less than workload structure unless the router can
  exploit session locality or prefix reuse.

## Product Direction

The benchmark should also become a practical serving-team tool. A future CLI
could expose:

- `acb doctor` for vLLM/OpenAI-compatible readiness checks.
- `acb run` and `acb sweep` for reproducible experiment execution.
- `acb profile-traffic` for redacted production trace analysis.
- `acb recommend` for SLO/cost-aware policy selection.
- `acb bundle` for evidence sealing and review.
- `acb report --format html` for shareable decision reports.

## Paper Direction

The strongest paper frame is a measurement paper:

> When is adapter specialization worth its KV-cache footprint?

The paper should lead with reset-isolated serving evidence and capacity
frontiers, not broad model-quality claims. Quality metrics should be framed as
controlled task proxies unless validated against stronger external or human
scored benchmarks.

## Long-Term Roadmap

1. **Evidence substrate:** lifecycle state, resumability, provenance bundles,
   confidence tables, and release-tier discipline.
2. **Regime experiments:** workload-shape sweeps, policy regret, prefix-cache
   disabled baselines, capacity frontier expansion, and training-budget parity.
3. **Decision tooling:** traffic profiler, recommendation report, run registry,
   and serving readiness doctor.
4. **Backend breadth:** vLLM first, then SGLang/TGI/OpenAI-compatible metrics
   adapters where feasible.
5. **Operational loop:** continuous benchmarking, cost accounting, GPU TTL
   safety, sealed release bundles, and drift/regression alerts.

## Long-Term Success

The project succeeds if a serving team can bring its own traffic trace and get a
defensible answer about adapter strategy, hardware pressure, cache sensitivity,
and claim confidence without relying on undocumented notebook work or fragile
one-off GPU runs.
