# Reports

Benchmark reports, figures, and CSV tables are generated artifacts.

Run:

```bash
uv run python -m adapter_cache_bench.analysis.report --runs-dir artifacts/runs
uv run python -m adapter_cache_bench.bench.compare --runs-dir artifacts/runs
uv run python -m adapter_cache_bench.analysis.pareto --runs-dir artifacts/runs
uv run python -m adapter_cache_bench.analysis.slo --runs-dir artifacts/runs
```

Outputs are written under `reports/` and ignored by git so the public repo does
not present a synthetic mock run as a standing research result.
