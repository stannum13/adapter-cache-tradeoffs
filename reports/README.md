# Reports

Benchmark reports, figures, and CSV tables are generated artifacts.

Run:

```bash
uv run python -m specialization_cache_frontier.analysis.report --runs-dir artifacts/runs
uv run python -m specialization_cache_frontier.bench.compare --runs-dir artifacts/runs
uv run python -m specialization_cache_frontier.analysis.pareto --runs-dir artifacts/runs
uv run python -m specialization_cache_frontier.analysis.slo --runs-dir artifacts/runs
```

Outputs are written under `reports/` and ignored by git so the public repo does
not present a synthetic mock run as a standing research result.
