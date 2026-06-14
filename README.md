# Adapter Cache Bench

Specialist adapters improve quality, but every routing decision is also a cache
decision. This repo is a reproducible cache/routing benchmark harness for
testing when semantic routing, sticky routing, cache-aware routing, multitask
adapters, and activated-style adapters help or hurt under shared-prefix
workloads.

It does not claim model-quality results from the mock backend. The mock path is
for systems sanity: prefix reuse, cache fragmentation, memory pressure, routing
policy behavior, and latency SLO accounting. Real quality evidence should come
from file-backed eval sets run through a served causal transformer, such as the
optional vLLM path.

## Core question

When is model/adaptor specialization worth its KV-cache footprint?

The thesis is simple: specialist adapters can improve task quality, but standard
LoRA-style serving often namespaces prefix cache entries by adapter identity. In
shared-prefix workloads, the same long document can be cached once per adapter
instead of once per trust group. A serving system should jointly optimize
quality, cache locality, latency SLOs, memory footprint, and tenant isolation.

## Why naive semantic routing can hurt

Semantic routing sends QA to a QA adapter, JSON extraction to a JSON adapter,
and code/documentation tasks to a code adapter. That is locally sensible, but if
the prefix cache key includes the adapter, a workload that repeatedly uses the
same document may fragment into multiple adapter-specific copies.

The benchmark makes that tradeoff measurable:

- `standard_lora`: adapter identity is part of the prefix cache key.
- `base_shared`: optimistic base-model sharing baseline.
- `activated_lora`: tokens before an invocation marker can be shared.
- `copy_on_write`: shared base prefix plus adapter-specific deltas in the
  simulator.

## Quickstart

```bash
uv sync --extra dev
make check
make small
```

The default path runs on CPU and does not require internet after dependencies
are installed.

Each benchmark run writes:

- `artifacts/runs/{run_id}/requests.jsonl`
- `artifacts/runs/{run_id}/summary.json`
- `artifacts/runs/{run_id}/config_resolved.yaml`
- `artifacts/runs/{run_id}/manifest.json`

Generated reports, figures, and CSV tables are ignored by git. Recreate them
locally with:

```bash
make report
make compare
make pareto
make slo
```

## What is real vs simulated

Use the mock backend when you want deterministic cache and routing experiments:

```bash
make small
make matrix
uv run python -m adapter_cache_bench.bench.run_matrix \
  --config configs/benchmark/memory_pressure.yaml
uv run python -m adapter_cache_bench.bench.run_matrix \
  --config configs/benchmark/repeated.yaml
```

Use JSONL eval configs when you want task records with ground truth:

```bash
make validate-eval-large
make validate-source-eval
make source-eval
uv run python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/public_domain_eval_large.yaml
```

Use a real backend when you want model outputs. The local Hugging Face backend
loads a causal LM directly, while the vLLM/OpenAI-compatible backend sends
`/chat/completions` requests and scores responses with the same task metrics
used by the benchmark:

```bash
make transformers-source-eval
make vllm-source-eval
make vllm-source-eval-l4-qwen
make vllm-source-eval-lora-qwen
uv run python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/vllm_example.yaml
uv run python -m adapter_cache_bench.bench.run_concurrent \
  --config configs/benchmark/heldout_xlarge_sft_eval_vllm_lora_trained_qwen15b_concurrent.yaml
```

See [docs/vllm.md](docs/vllm.md) for the optional serving flow.
See [docs/model_backends.md](docs/model_backends.md) for backend options.
See [docs/gcloud_vllm.md](docs/gcloud_vllm.md) for a GPU/vLLM runbook.

The LoRA vLLM path expects the server to expose adapter model names such as
`qa-lora`, `json-lora`, `summary-lora`, and `code-lora`. The included Qwen LoRA
overlay is a serving smoke path; replace it with task-trained adapters before
claiming specialist quality gains.

See [docs/eval_datasets.md](docs/eval_datasets.md) for the JSONL schema.
See [docs/real_eval_results.md](docs/real_eval_results.md) for a real vLLM
snapshot with trained Qwen LoRA adapters.
See [docs/release_checklist.md](docs/release_checklist.md) before publishing.

## Workloads

- `shared_doc_qa`: many questions over the same long document.
- `mixed_tasks_same_doc`: QA, JSON extraction, summarization, and code/doc tasks
  over the same document.
- `agent_session`: multi-turn sessions with growing history and repeated tool
  traces.
- `low_overlap_control`: random prompts with little shared prefix.
- `prompt_layout_ablation`: compares task-before-document against
  document-before-task layouts to test late-specialization locality.
- `jsonl_eval`: file-backed records for replacing synthetic prompts with real
  evaluation sets.

## Router policies

- `random`: baseline traffic spread.
- `semantic`: route by task type to the expected specialist.
- `multitask`: route every task to a shared multitask adapter.
- `sticky_session`: keep a session on the same adapter when compatible.
- `cache_aware`: combine quality prior, estimated cached tokens, queue penalty,
  switch penalty, isolation penalty, and cold-adapter penalty.
- `oracle`: simulator upper bound using known ground truth and backend quality.

## Repo structure

- `configs/`: benchmark, router, cache, and workload YAMLs.
- `data/eval/`: small public-domain style JSONL fixtures.
- `src/adapter_cache_bench/cache/`: whitespace tokenizer and block
  prefix-cache simulators.
- `src/adapter_cache_bench/routing/`: router policies.
- `src/adapter_cache_bench/backends/`: mock, local Hugging Face, and
  OpenAI-compatible/vLLM clients.
- `src/adapter_cache_bench/bench/`: workload, matrix, metric, and
  comparison runners.
- `src/adapter_cache_bench/analysis/`: report, plot, SLO, and Pareto
  helpers.
- `src/adapter_cache_bench/physical_ai_analogue/`: lightweight
  scene-cache analogy for VLA/world-model serving.

## Local verification

```bash
uv run pytest tests -q
uv run ruff check .
uv run ruff format . --check
```

The GitHub Actions workflow runs the same CPU-only checks. GPU or serving tests
must stay optional and are skipped unless explicitly enabled.

## Current limitations

- Whitespace tokenization approximates prefix blocks; it is not a tokenizer
  match for any serving engine.
- Mock quality uses deterministic priors and noise; it is useful for regression
  testing, not model evaluation.
- Cache memory accounting is token-based, not byte-accurate KV allocation.
- The activated-LoRA and copy-on-write paths are simulators, not vLLM kernel
  implementations.
- The included vLLM LoRA config proves real adapter serving mechanics, but its
  public smoke adapter is not trained for this benchmark's QA/JSON/summary/code
  tasks.
- The included JSONL eval fixtures are intentionally small and public-domain
  style. Replace them before making research claims.

## Physical AI analogue

The same cache-specialization problem appears in VLA serving. Text prefix tokens
map to repeated visual/proprioceptive scene tokens; KV cache maps to a
world-state cache; LoRA adapters map to skill or embodiment adapters; TTFT and
goodput map to control latency and success-rate-adjusted control Hz.

See
[src/adapter_cache_bench/physical_ai_analogue/README.md](src/adapter_cache_bench/physical_ai_analogue/README.md).

## License

MIT. See [LICENSE](LICENSE).
