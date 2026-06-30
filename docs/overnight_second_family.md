# Overnight Second-Family Loop

This loop is for the remaining cross-family evidence gap: run the same source-backed
external workload on Qwen and one non-Qwen causal-transformer family with matching
task LoRAs, then regenerate the public reports and readiness table.

The script is stateful. It writes all operator state under
`artifacts/overnight/<timestamp>/`:

- `state.json`: current phase, status, selected models, and git commit.
- `events.jsonl`: one JSON event per phase attempt.
- `commands/*.log`: stdout and stderr for each phase.
- `markers/*.done`: idempotency markers used to skip completed phases on resume.
- `configs/*.yaml`: generated single-family sweep configs.
- `summary.md`: morning-readable status and readiness output.

## Default Plan

Run:

```bash
make overnight-second-family
```

By default it:

1. Validates the repo, external fixture, Ruff, formatting, Docker, and GPU visibility.
2. Builds an SFT split from `configs/benchmark/external_eval_vllm_template.yaml`.
3. Trains TinyLlama task adapters if `artifacts/adapters/tinyllama11b-*` are absent.
4. Trains Qwen 1.5B task adapters if `artifacts/adapters/qwen15b-*` are absent.
5. Serves TinyLlama adapters in vLLM and runs a `model-family-vllm-streaming` sweep.
6. Serves Qwen adapters in vLLM and runs the same sweep.
7. Regenerates reports, figures, adapter metrics, and research readiness output.
8. Runs `uv run ruff check .`, `uv run ruff format . --check`, and
   `uv run pytest tests -q`.

## Dry Run

Validate the generated state layout and commands without touching Docker, GPU, or
model downloads:

```bash
DRY_RUN=1 OVERNIGHT_RUN_DIR=/tmp/acb-overnight-dry-run make overnight-second-family
```

## Useful Overrides

```bash
SECOND_BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
SECOND_ALIAS=tinyllama11b
SECOND_PREFIX=tinyllama11b
QWEN_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
QWEN_PREFIX=qwen15b
OVERNIGHT_REQUEST_COUNT=500
OVERNIGHT_SEEDS_CSV=17,23,31
OVERNIGHT_STRATEGIES_CSV=specialists,multitask
MAX_STEPS=40
MULTITASK_MAX_STEPS=80
LOAD_IN_4BIT=1
```

Set `TRAIN_QWEN_IF_MISSING=0` if Qwen adapters must already exist and a missing
bundle should fail early.

For VM cleanup, pass shell hooks:

```bash
ON_SUCCESS_COMMAND='gcloud compute instances stop ...'
ON_FAILURE_COMMAND='gcloud compute instances stop ...'
```

Use `ON_FAILURE_COMMAND` only when logs and Docker state do not need to remain live
for debugging.

## Resume

Re-run with the same `OVERNIGHT_RUN_DIR`. Completed phases are skipped by marker:

```bash
OVERNIGHT_RUN_DIR=artifacts/overnight/20260620-010000 make overnight-second-family
```

Delete a single marker to replay one phase:

```bash
rm artifacts/overnight/20260620-010000/markers/run_second_family.done
OVERNIGHT_RUN_DIR=artifacts/overnight/20260620-010000 make overnight-second-family
```

For future matrix-style overnight work, prefer the resumable sweep flags in
[sweep_operations.md](sweep_operations.md). Run a `--dry-run` with budget gates
before launching GPU work, then use a stable `--sweep-name` and `--resume` so
interrupted child runs are recovered instead of duplicated.

## Interpretation

The loop closes the readiness gap only if `research_readiness.md` reports
`multi_model_comparison` as `ok`. That means at least two served model-family
aliases were observed in `model-family-vllm` run summaries with complete adapter
name mappings. It does not by itself prove the paper claim; inspect the regenerated
tables and figures for effect direction, confidence intervals, and any quality
regressions before strengthening the claim language.
