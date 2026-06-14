.PHONY: sync test lint format check small matrix report compare pareto slo validate-eval validate-eval-large validate-eval-xlarge validate-source-eval source-eval transformers-source-eval vllm-example vllm-source-eval vllm-source-eval-l4-qwen vllm-source-eval-l4-qwen15b vllm-source-eval-lora-qwen vllm-source-eval-lora-trained-qwen15b vllm-heldout-qwen15b vllm-heldout-lora-trained-qwen15b vllm-heldout-lora-trained-qwen15b-standard vllm-heldout-lora-multitask-qwen15b vllm-heldout-xlarge-qwen15b vllm-heldout-xlarge-lora-trained-qwen15b vllm-heldout-xlarge-lora-multitask-qwen15b vllm-heldout-xlarge-qwen15b-concurrent vllm-heldout-xlarge-lora-trained-qwen15b-concurrent vllm-heldout-xlarge-lora-multitask-qwen15b-concurrent vllm-overnight-frontier vllm-heldout-trained-matrix-qwen15b vllm-heldout-trained-repeated-qwen15b reproduce-mock

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

validate-eval-xlarge:
	uv run python -m adapter_cache_bench.workloads.validate_dataset --config configs/benchmark/public_domain_eval_xlarge.yaml

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

vllm-heldout-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_sft_eval_vllm_qwen15b.yaml

vllm-heldout-lora-trained-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_sft_eval_vllm_lora_trained_qwen15b.yaml

vllm-heldout-lora-trained-qwen15b-standard:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_sft_eval_vllm_lora_trained_qwen15b_standard.yaml

vllm-heldout-lora-multitask-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_sft_eval_vllm_lora_multitask_qwen15b.yaml

vllm-heldout-xlarge-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_xlarge_sft_eval_vllm_qwen15b.yaml

vllm-heldout-xlarge-lora-trained-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_trained_qwen15b.yaml

vllm-heldout-xlarge-lora-multitask-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_multitask_qwen15b.yaml

vllm-heldout-xlarge-qwen15b-concurrent:
	uv run python -m adapter_cache_bench.bench.run_concurrent --config configs/benchmark/heldout_xlarge_sft_eval_vllm_qwen15b_concurrent.yaml

vllm-heldout-xlarge-lora-trained-qwen15b-concurrent:
	uv run python -m adapter_cache_bench.bench.run_concurrent --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_trained_qwen15b_concurrent.yaml

vllm-heldout-xlarge-lora-multitask-qwen15b-concurrent:
	uv run python -m adapter_cache_bench.bench.run_concurrent --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_multitask_qwen15b_concurrent.yaml

vllm-overnight-frontier:
	uv run python -m adapter_cache_bench.bench.run_concurrency_sweep --config configs/benchmark/overnight_frontier_vllm.yaml

vllm-heldout-trained-matrix-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/heldout_trained_matrix_vllm_qwen15b.yaml

vllm-heldout-trained-repeated-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/heldout_trained_repeated_vllm_qwen15b.yaml

reproduce-mock: matrix
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/memory_pressure.yaml
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/repeated.yaml
	$(MAKE) source-eval
	$(MAKE) report compare pareto slo
