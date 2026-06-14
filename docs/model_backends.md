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
- `backend.adapter_model_names` for vLLM LoRA modules served as model names
- `backend.temperature`
- `backend.extra_body`

Run the source eval against vLLM:

```bash
make vllm-source-eval
```

For real vLLM LoRA serving, start vLLM with `--enable-lora --lora-modules`
and map benchmark adapter IDs to served model names:

```yaml
backend:
  model: Qwen/Qwen2.5-3B-Instruct
  adapter_model_names:
    qa: qa-lora
    json: json-lora
```

When `adapter_model_names` is not set, the backend keeps adapter metadata in
`extra_body.adapter` for custom OpenAI-compatible servers that use a non-vLLM
adapter convention.

For a GPU VM setup, see [gcloud_vllm.md](gcloud_vllm.md).
