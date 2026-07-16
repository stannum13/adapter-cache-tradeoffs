# Limitations

This document is the current public evidence boundary for Adapter Cache
Tradeoffs.

## Current Status

E001 is preregistered but has not been canonically run on SGLang. The repository
currently supports local smoke verification, dry-run budget checks, historical
evidence review, and OpenAI-compatible serving integrations.

## Measurement Limits

- `simulated_cached_prefix_tokens` and `simulated_cached_prompt_tokens` are
  benchmark-side estimates. They are not server prefix-cache counters.
- `server_prefix_cache_*` fields are server-level counters when the runtime
  exposes them. They are not per-request adapter-cache measurements.
- The mock backend is deterministic systems scaffolding. It is not model
  quality evidence.
- Whitespace tokenization approximates prefix blocks and memory footprint. It
  does not match exact runtime tokenization or byte-accurate KV allocation.

## Evidence Limits

- Historical vLLM results remain useful prior evidence, but some predate the
  current run lifecycle contract requiring `status.json`.
- New public claims should come from current-contract runs with raw request
  records, resolved configs, manifests, summaries, and deterministic analysis.
- The E001 SGLang run, Vidur policy screening, and vLLM replication are pending.
- Quality scores from fixture-style evals are bounded to those fixtures and
  should not be described as standard public benchmark performance.

## Scope Limits

- This repository does not vendor SGLang, Vidur, or vLLM.
- It does not claim production serving recommendations across all models,
  adapter ranks, context lengths, GPUs, or traffic mixes.
- Activated-LoRA-style cache behavior is simulator-side unless a serving runtime
  exposes matching kernel/cache-key behavior.
- Cost and power frontiers are not measured here.
