.PHONY: sync test lint format check small matrix benchmark-v0-mock benchmark-v0-csv build-external-eval report release-report whitepaper-figure large-model-figures adapter-metrics model-family-summary claim-evidence policy-regret capacity-frontier research-readiness compare pareto slo validate-eval validate-eval-large validate-eval-xlarge validate-source-eval validate-source-eval-expanded validate-external-eval source-eval source-eval-expanded transformers-source-eval train-qwen7b-adapters train-qwen7b-adapters-seed23 train-qwen7b-adapters-seed31 overnight-second-family vllm-example vllm-source-eval vllm-source-eval-l4-qwen vllm-source-eval-l4-qwen15b vllm-source-eval-l4-qwen7b vllm-source-eval-expanded-qwen7b vllm-source-eval-lora-qwen vllm-source-eval-lora-trained-qwen15b vllm-source-eval-lora-trained-qwen7b vllm-source-eval-lora-multitask-qwen7b vllm-source-eval-expanded-lora-trained-qwen7b vllm-source-eval-expanded-lora-multitask-qwen7b vllm-source-eval-lora-trained-qwen7b-seed23 vllm-source-eval-lora-multitask-qwen7b-seed23 vllm-source-eval-lora-trained-qwen7b-seed31 vllm-source-eval-lora-multitask-qwen7b-seed31 vllm-external-eval vllm-model-family vllm-large-model-pilot vllm-large-model-confidence vllm-large-model-confidence-reset vllm-large-model vllm-heldout-qwen15b vllm-heldout-lora-trained-qwen15b vllm-heldout-lora-trained-qwen15b-standard vllm-heldout-lora-multitask-qwen15b vllm-heldout-xlarge-qwen15b vllm-heldout-xlarge-lora-trained-qwen15b vllm-heldout-xlarge-lora-multitask-qwen15b vllm-heldout-xlarge-qwen15b-concurrent vllm-heldout-xlarge-lora-trained-qwen15b-concurrent vllm-heldout-xlarge-lora-multitask-qwen15b-concurrent vllm-heldout-xlarge-qwen7b vllm-heldout-xlarge-lora-trained-qwen7b vllm-heldout-xlarge-lora-multitask-qwen7b vllm-heldout-xlarge-qwen7b-concurrent vllm-heldout-xlarge-lora-trained-qwen7b-concurrent vllm-heldout-xlarge-lora-multitask-qwen7b-concurrent vllm-overnight-frontier vllm-overnight-frontier-streaming vllm-exhaustive-layout vllm-exhaustive-overlap vllm-exhaustive-overlap-reset vllm-exhaustive-adapter-count vllm-exhaustive-adapter-count-reset vllm-exhaustive-tenant-isolation vllm-exhaustive-confidence vllm-exhaustive-all vllm-heldout-trained-matrix-qwen15b vllm-heldout-trained-repeated-qwen15b reproduce-mock

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

benchmark-v0-mock:
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/benchmark_v0_mock.yaml

benchmark-v0-csv:
	uv run python -m adapter_cache_bench.analysis.benchmark_v0 --runs-dir artifacts/runs --output-csv data/results/benchmark_v0_mock.csv

build-external-eval:
	uv run python -m adapter_cache_bench.workloads.build_external_eval --output data/eval/external_public_domain_eval.jsonl

report:
	uv run python -m adapter_cache_bench.analysis.report --runs-dir artifacts/runs

release-report:
	uv run python -m adapter_cache_bench.analysis.report --runs-dir artifacts/runs --report-path docs/release_report.md

whitepaper-figure:
	uv run python -m adapter_cache_bench.analysis.whitepaper_visual --runs-dir artifacts/runs --output-dir docs/figures

large-model-figures:
	uv run python -m adapter_cache_bench.analysis.large_model_plots --runs-dir artifacts/runs

adapter-metrics:
	uv run python -m adapter_cache_bench.analysis.adapter_cache_metrics --runs-dir artifacts/runs --output reports/tables/adapter_cache_metrics.csv

model-family-summary:
	uv run python -m adapter_cache_bench.analysis.model_family --runs-dir artifacts/runs --output reports/tables/model_family_summary.csv

claim-evidence:
	uv run python -m adapter_cache_bench.analysis.claim_tables --runs-dir artifacts/runs --output reports/tables/claim_evidence.csv

policy-regret:
	uv run python -m adapter_cache_bench.analysis.policy_regret --runs-dir artifacts/runs --output reports/tables/policy_regret.csv

capacity-frontier:
	uv run python -m adapter_cache_bench.analysis.capacity_frontier --output-csv reports/tables/capacity_frontier.csv

research-readiness:
	uv run python -m adapter_cache_bench.analysis.research_readiness --runs-dir artifacts/runs

.PHONY: evidence-bundle
evidence-bundle:
	uv run python -m adapter_cache_bench.analysis.evidence_bundle --bundle-name $(or $(BUNDLE),latest) $(if $(OUTPUT),--output-dir $(OUTPUT),) $(foreach run,$(RUNS),--run $(run)) $(foreach pattern,$(RUN_GLOBS),--run-glob $(pattern)) $(foreach report,$(REPORTS),--report $(report)) $(foreach figure,$(FIGURES),--figure $(figure))

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

validate-source-eval-expanded:
	uv run python -m adapter_cache_bench.workloads.validate_dataset --config configs/benchmark/source_eval_expanded.yaml --min-records 200 --require-tasks qa,json,summary,code --require-layouts document_before_instruction,instruction_before_document --balanced-tasks --min-shared-prefix-groups 15 --require-tenant-fields --require-source-fields --require-public-domain-license

validate-external-eval:
	uv run python -m adapter_cache_bench.workloads.validate_dataset --config configs/benchmark/external_eval_vllm_template.yaml --min-records 500 --require-tasks qa,json,summary,code --require-layouts document_before_instruction,instruction_before_document --balanced-tasks --min-shared-prefix-groups 25 --require-tenant-fields --require-source-fields --require-public-domain-license

source-eval:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval.yaml

source-eval-expanded:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_expanded.yaml

transformers-source-eval:
	uv run --extra real python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_transformers.yaml

train-qwen7b-adapters:
	BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SFT_DIR=artifacts/sft/public_domain_xlarge OUTPUT_PREFIX=qwen7b LOAD_IN_4BIT=1 MAX_STEPS=40 MULTITASK_MAX_STEPS=80 MAX_LENGTH=768 ./scripts/train_qwen15b_task_adapters.sh

train-qwen7b-adapters-seed23:
	BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SFT_DIR=artifacts/sft/public_domain_xlarge OUTPUT_PREFIX=qwen7b-seed23 TRAIN_SEED=23 LOAD_IN_4BIT=1 MAX_STEPS=40 MULTITASK_MAX_STEPS=80 MAX_LENGTH=768 ./scripts/train_qwen15b_task_adapters.sh

train-qwen7b-adapters-seed31:
	BASE_MODEL=Qwen/Qwen2.5-7B-Instruct SFT_DIR=artifacts/sft/public_domain_xlarge OUTPUT_PREFIX=qwen7b-seed31 TRAIN_SEED=31 LOAD_IN_4BIT=1 MAX_STEPS=40 MULTITASK_MAX_STEPS=80 MAX_LENGTH=768 ./scripts/train_qwen15b_task_adapters.sh

overnight-second-family:
	./scripts/overnight_second_family_loop.sh

vllm-example:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/vllm_example.yaml

vllm-source-eval:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml

vllm-source-eval-l4-qwen:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_l4_qwen.yaml

vllm-source-eval-l4-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_l4_qwen15b.yaml

vllm-source-eval-l4-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_qwen7b.yaml

vllm-source-eval-expanded-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_expanded_vllm_qwen7b.yaml

vllm-source-eval-lora-qwen:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_qwen.yaml

vllm-source-eval-lora-trained-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_trained_qwen15b.yaml

vllm-source-eval-lora-trained-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_trained_qwen7b.yaml

vllm-source-eval-lora-multitask-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_multitask_qwen7b.yaml

vllm-source-eval-expanded-lora-trained-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_expanded_vllm_lora_trained_qwen7b.yaml

vllm-source-eval-expanded-lora-multitask-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_expanded_vllm_lora_multitask_qwen7b.yaml

vllm-source-eval-lora-trained-qwen7b-seed23:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_trained_qwen7b_seed23.yaml

vllm-source-eval-lora-multitask-qwen7b-seed23:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_multitask_qwen7b_seed23.yaml

vllm-source-eval-lora-trained-qwen7b-seed31:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_trained_qwen7b_seed31.yaml

vllm-source-eval-lora-multitask-qwen7b-seed31:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/source_eval_vllm.yaml configs/benchmark/source_eval_vllm_lora_multitask_qwen7b_seed31.yaml

vllm-external-eval:
	uv run python -m adapter_cache_bench.bench.run_concurrent --config configs/benchmark/external_eval_vllm_template.yaml

vllm-model-family:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/model_family_vllm_template.yaml

vllm-large-model-pilot:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/large_model_pilot_vllm.yaml

vllm-large-model-confidence:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/large_model_overlap_confidence_vllm.yaml

vllm-large-model-confidence-reset:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/large_model_overlap_confidence_vllm.yaml configs/benchmark/gcloud_7b_reset_template.yaml

vllm-large-model:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/large_model_vllm_template.yaml

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

vllm-heldout-xlarge-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_xlarge_sft_eval_vllm_qwen7b.yaml

vllm-heldout-xlarge-lora-trained-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_trained_qwen7b.yaml

vllm-heldout-xlarge-lora-multitask-qwen7b:
	uv run python -m adapter_cache_bench.bench.run_workload --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_multitask_qwen7b.yaml

vllm-heldout-xlarge-qwen7b-concurrent:
	uv run python -m adapter_cache_bench.bench.run_concurrent --config configs/benchmark/heldout_xlarge_sft_eval_vllm_qwen7b_concurrent.yaml

vllm-heldout-xlarge-lora-trained-qwen7b-concurrent:
	uv run python -m adapter_cache_bench.bench.run_concurrent --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_trained_qwen7b_concurrent.yaml

vllm-heldout-xlarge-lora-multitask-qwen7b-concurrent:
	uv run python -m adapter_cache_bench.bench.run_concurrent --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_multitask_qwen7b_concurrent.yaml

vllm-overnight-frontier:
	uv run python -m adapter_cache_bench.bench.run_concurrency_sweep --config configs/benchmark/overnight_frontier_vllm.yaml

vllm-overnight-frontier-streaming:
	uv run python -m adapter_cache_bench.bench.run_concurrency_sweep --config configs/benchmark/overnight_frontier_vllm_streaming.yaml

vllm-exhaustive-layout:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/exhaustive_layout_vllm_streaming.yaml

vllm-exhaustive-overlap:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/exhaustive_overlap_vllm_streaming.yaml

vllm-exhaustive-overlap-reset:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/exhaustive_overlap_vllm_streaming.yaml configs/benchmark/local_vllm_reset.yaml

vllm-exhaustive-adapter-count:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/exhaustive_adapter_count_vllm_streaming.yaml

vllm-exhaustive-adapter-count-reset:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/exhaustive_adapter_count_vllm_streaming.yaml configs/benchmark/local_vllm_reset.yaml

vllm-exhaustive-tenant-isolation:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/exhaustive_tenant_isolation_vllm_streaming.yaml

vllm-exhaustive-confidence:
	uv run python -m adapter_cache_bench.bench.run_exhaustive_sweep --config configs/benchmark/exhaustive_confidence_vllm_streaming.yaml

vllm-exhaustive-all: vllm-exhaustive-layout vllm-exhaustive-overlap vllm-exhaustive-adapter-count vllm-exhaustive-tenant-isolation vllm-exhaustive-confidence

vllm-heldout-trained-matrix-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/heldout_trained_matrix_vllm_qwen15b.yaml

vllm-heldout-trained-repeated-qwen15b:
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/heldout_trained_repeated_vllm_qwen15b.yaml

reproduce-mock: matrix
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/memory_pressure.yaml
	uv run python -m adapter_cache_bench.bench.run_matrix --config configs/benchmark/repeated.yaml
	$(MAKE) source-eval
	$(MAKE) report compare pareto slo
