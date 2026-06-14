.PHONY: sync test lint format check small matrix report compare pareto slo validate-eval validate-eval-large validate-source-eval source-eval transformers-source-eval vllm-example vllm-source-eval vllm-source-eval-l4-qwen vllm-source-eval-l4-qwen15b vllm-source-eval-lora-qwen vllm-source-eval-lora-trained-qwen15b reproduce-mock

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

validate-source-eval:
	uv run python -m adapter_cache_bench.workloads.validate_dataset --config configs/benchmark/source_eval.yaml

source-eval:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval.yaml

transformers-source-eval:
	uv run --extra real python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_transformers.yaml

vllm-example:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/vllm_example.yaml

vllm-source-eval:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml

vllm-source-eval-l4-qwen:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_l4_qwen.yaml

vllm-source-eval-l4-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_l4_qwen15b.yaml

vllm-source-eval-lora-qwen:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_qwen.yaml

vllm-source-eval-lora-trained-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_trained_qwen15b.yaml

reproduce-mock: matrix
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/memory_pressure.yaml
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/repeated.yaml
	$(MAKE) source-eval
	$(MAKE) report compare pareto slo
