# Reports

Benchmark reports, figures, and CSV tables are generated artifacts.

Run:

```bash
uv run python -m adapter_cache_bench.analysis.report --runs-dir artifacts/runs
uv run python -m adapter_cache_bench.bench.compare --runs-dir artifacts/runs
uv run python -m adapter_cache_bench.analysis.pareto --runs-dir artifacts/runs
uv run python -m adapter_cache_bench.analysis.slo --runs-dir artifacts/runs
uv run python -m adapter_cache_bench.analysis.policy_regret --runs-dir artifacts/runs --output reports/tables/policy_regret.csv
uv run python -m adapter_cache_bench.analysis.regime_science --runs-dir artifacts/runs --output reports/figures/regime_policy_failure_map.png
```

Outputs are written under `reports/` and ignored by git so the public repo does
not present a synthetic mock run as a standing research result.

`policy_regret.csv` is a regime-science table: it groups comparable runs by
workload and non-policy sweep dimensions, ranks policies by quality-adjusted
goodput, and reports regret against the best observed policy in each regime.

`regime_policy_failure_map.png` is generated only when regime workloads are
present. It visualizes relative regret by workload regime and policy using the
same dark scientific figure style as the rest of the report.
