# Contributing

This repository is designed to stay CPU-testable and reproducible.

## Local checks

```bash
uv sync --extra dev
make check
```

## Eval data

File-backed eval workloads should be JSONL or YAML and must include:

- `task_type`
- `document`
- `question` or `prompt`
- `ground_truth`
- `expected_adapter`

Validate the bundled eval fixture with:

```bash
make validate-eval
```

## GPU and serving integrations

Unit tests must not require a GPU, internet access, or a running vLLM server.
Real serving tests should stay behind explicit environment gates such as
`RUN_VLLM_TESTS=1`.

## Benchmark artifacts

Every benchmark run should write:

- `requests.jsonl`
- `summary.json`
- `config_resolved.yaml`
- `manifest.json`

Keep generated artifacts small and reproducible before committing them.
