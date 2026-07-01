# Short-Term Concept: Auditable Adapter-Serving Evidence

## Purpose

The next short-term goal is to turn `adapter-cache-tradeoffs` from a working
benchmark harness into an auditable evidence machine. The project already
generates useful vLLM and mock-serving results, but the next credibility jump
requires stateful run lifecycle, resumability, stronger uncertainty tables, and
frozen provenance bundles.

The short-term product should answer:

> Can this result be trusted, resumed, audited, and compared without rerunning
> an expensive GPU experiment from scratch?

## Scope

This phase is about evidence quality, not new headline claims. It prioritizes
the mechanics needed before another large overnight run:

- Every run writes start-time artifacts before the first request.
- Partial failures preserve useful request rows and failure status.
- Sweeps can resume missing or failed children.
- Manifests include enough runtime/backend metadata to explain the run later.
- Claim-critical reports include repeated-run uncertainty and paired deltas.
- Evidence bundles freeze selected run outputs, configs, reports, and hashes.

## Near-Term User

The primary user is the repo maintainer or a collaborating systems researcher
running GPU experiments. A secondary user is an external reviewer who wants to
verify that public claims are backed by reproducible artifacts.

## Non-Goals

- Do not broaden claims about universal LoRA quality.
- Do not make GPU or internet access required for unit tests.
- Do not commit raw run artifacts, adapters, caches, cloud keys, or virtualenvs.
- Do not build a full public product CLI before run evidence is reliable.

## Success Criteria

- A failed run still has `config_resolved.yaml`, `manifest.json`, `status.json`,
  and any completed request rows.
- A completed run records backend identity, served model, adapter names, stream
  mode, concurrency, metric scope, and git metadata.
- A selected set of run IDs can be sealed into a machine-readable evidence
  bundle manifest with SHA256 hashes.
- Reports can surface mean, 95% confidence intervals, and paired deltas for
  repeated specialist/multitask comparisons.
- The next overnight run can be resumed or diagnosed from machine-readable
  state instead of shell transcript archaeology.

## First Build Slice

1. Run lifecycle and partial-failure artifacts.
2. Evidence bundle manifest generator.
3. Claim evidence and paired-delta tables.
4. Resumable sweep checkpointing.
5. Quickstart and examples once the artifact contract is stable.

## Decision Rule

Prefer work that makes future expensive evidence harder to lose, easier to
audit, or safer to interpret. Defer purely presentational work unless it exposes
missing evidence or prevents overclaiming.
