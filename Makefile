.PHONY: sync test lint format check small matrix report compare pareto slo validate-eval validate-eval-large vllm-example reproduce-mock

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
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/small.yaml

matrix:
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/full.yaml

report:
	uv run python -m adapter_cache_bench.analysis.report --runs-dir artifacts/runs

compare:
	uv run python -m adapter_cache_bench.bench.compare --runs-dir artifacts/runs

pareto:
	uv run python -m adapter_cache_bench.analysis.pareto --runs-dir artifacts/runs

slo:
	uv run python -m adapter_cache_bench.analysis.slo --runs-dir artifacts/runs

validate-eval:
	uv run python -m adapter_cache_bench.workloads.validate_dataset --config configs/benchmark/public_domain_eval.yaml

validate-eval-large:
	uv run python -m adapter_cache_bench.workloads.validate_dataset --config configs/benchmark/public_domain_eval_large.yaml

vllm-example:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/vllm_example.yaml

reproduce-mock: matrix
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/memory_pressure.yaml
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/repeated.yaml
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/public_domain_eval_large.yaml
	$(MAKE) report compare pareto slo
