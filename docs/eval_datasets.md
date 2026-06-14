# Eval datasets

`jsonl_eval` is the path for replacing synthetic prompts with task records that
have ground truth.

Each line is one JSON object:

```json
{
  "request_id": "eval-qa-001",
  "document_id": "weather-log-1901",
  "task_type": "qa",
  "document": "shared document text...",
  "question": "Who observed the station notes?",
  "ground_truth": "ada",
  "expected_adapter": "qa",
  "prompt_layout": "document_before_instruction",
  "requires_json": false
}
```

Required fields:

- `request_id`: stable unique ID.
- `document_id`: shared-prefix grouping key.
- `task_type`: `qa`, `json`, `summary`, or `code`.
- `document`: text used as the shared prefix body.
- `question`: task invocation text.
- `ground_truth`: string for QA/summary, object for JSON, or an object with
  `tests` for code-style checks.
- `expected_adapter`: adapter label used by semantic routing and scoring.

Optional fields:

- `prompt_layout`: `document_before_instruction` or
  `instruction_before_document`.
- `requires_json`: whether the task expects JSON output.

Validate a dataset-backed config before running:

```bash
uv run python -m adapter_cache_bench.workloads.validate_dataset \
  --config configs/benchmark/public_domain_eval_large.yaml
```

Run it with the mock backend for cache/routing sanity:

```bash
uv run python -m adapter_cache_bench.bench.run_workload \
  --config configs/benchmark/public_domain_eval_large.yaml
```

Run the same shape through a served causal transformer by setting
`backend.kind: vllm` and configuring `backend.base_url`, `backend.model`, and
adapter routing metadata. See [vllm.md](vllm.md).
