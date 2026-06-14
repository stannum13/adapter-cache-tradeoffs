# vLLM and local model servers

The default benchmark path uses `MockBackend` and requires no GPU. The optional
`VLLMBackend` uses an OpenAI-compatible `/chat/completions` endpoint. For a
generic local server, set `backend.kind: openai_compatible`; for vLLM-specific
configs, use `backend.kind: vllm`.

## Start a Server

Example shape:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora
```

Adapter naming and LoRA registration depend on the vLLM version and serving
layout. Keep those choices in `backend.extra_body` or server-side model names.

See [model_backends.md](model_backends.md) for the backend matrix.

## Run an Optional Smoke Benchmark

```bash
RUN_VLLM_TESTS=1 uv run pytest tests/test_optional_integrations.py -q
make vllm-example
```

`vllm_example.yaml` is intentionally small. `source_eval_vllm.yaml` runs the
source-backed eval bundle against the same OpenAI-compatible endpoint:

```bash
make vllm-source-eval
```

Configure `backend.base_url`, `backend.model`, and adapter metadata for your
server. Use the mock backend for unit tests, CI, and CPU-only development. vLLM
responses are scored with the benchmark's task metrics (`qa`, `json`,
`summary`, and `code`) when `ground_truth` is present in the request record.

If no vLLM server is available, `configs/benchmark/source_eval_transformers.yaml`
runs the same benchmark harness through a local Hugging Face causal LM backend:

```bash
make transformers-source-eval
```

## Metrics

The backend records wall-clock request latency from the client. For production
serving studies, also scrape server-side Prometheus metrics with
`MetricsClient` and join them with `requests.jsonl` by run timestamp or request
metadata.
