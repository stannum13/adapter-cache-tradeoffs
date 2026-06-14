.PHONY: sync test lint format check small matrix report compare pareto slo

sync:
	uv sync --extra dev

test:
	uv run pytest tests -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: test lint
	uv run ruff format . --check

small:
	uv run python -m specialization_cache_frontier.bench.run_workload --config configs/benchmark/small.yaml

matrix:
	uv run python -m specialization_cache_frontier.bench.run_matrix --config configs/benchmark/full.yaml

report:
	uv run python -m specialization_cache_frontier.analysis.report --runs-dir artifacts/runs

compare:
	uv run python -m specialization_cache_frontier.bench.compare --runs-dir artifacts/runs

pareto:
	uv run python -m specialization_cache_frontier.analysis.pareto --runs-dir artifacts/runs

slo:
	uv run python -m specialization_cache_frontier.analysis.slo --runs-dir artifacts/runs
