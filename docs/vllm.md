# vLLM Integration

The default benchmark path uses `MockBackend` and requires no GPU. The optional
`VLLMBackend` uses an OpenAI-compatible `/chat/completions` endpoint.

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

## Run an Optional Smoke Benchmark

```bash
RUN_VLLM_TESTS=1 uv run pytest tests/test_optional_integrations.py -q
uv run python -m specialization_cache_frontier.bench.run_workload \
  --config configs/benchmark/vllm_example.yaml
```

`vllm_example.yaml` is intentionally small. Use the mock backend for unit tests,
CI, and CPU-only development.

## Metrics

The backend records wall-clock request latency from the client. For production
serving studies, also scrape server-side Prometheus metrics with
`MetricsClient` and join them with `requests.jsonl` by run timestamp or request
metadata.
