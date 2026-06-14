# Model backends

The benchmark has three backend tiers.

## Mock backend

`backend.kind: mock`

Use this for deterministic cache, routing, memory-pressure, and SLO experiments.
It does not produce real model quality.

```bash
make source-eval
```

## Local Hugging Face causal LM backend

`backend.kind: transformers`

Use this when you need a real local model-output run without a model server.
The default config uses a small CPU-capable causal LM and the source-backed eval
bundle:

```bash
make transformers-source-eval
```

This path uses optional dependencies:

```bash
uv sync --extra dev --extra real
```

## OpenAI-compatible local server backend

`backend.kind: vllm` or `backend.kind: openai_compatible`

Use this for vLLM and other local servers exposing `/v1/chat/completions`.
Configure:

- `backend.base_url`
- `backend.api_key`
- `backend.model`
- `backend.temperature`
- `backend.extra_body`

Run the source eval against vLLM:

```bash
make vllm-source-eval
```

The request payload keeps adapter metadata in `extra_body.adapter`, so server
deployments can map it to their LoRA or adapter-loading convention.

For a GPU VM setup, see [gcloud_vllm.md](gcloud_vllm.md).
