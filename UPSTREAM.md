# Upstream Pins

This repository is the experiment harness. E001 must keep serving-runtime changes in upstream substrates or focused forks, not vendored into this tree.

Pinned on 2026-07-17:

| Role | Repository | Pinned commit | Use in E001 |
| --- | --- | --- | --- |
| Primary serving substrate | `https://github.com/sgl-project/sglang.git` | `68f4de162d99e78742b3de3e17f3cd497db9e9a6` | Preferred runtime for canonical adapter-locality runs. Start with its OpenAI-compatible server path; fork only if scheduler/cache internals must change. |
| Low-cost policy screening | `https://github.com/microsoft/vidur.git` | `8383d2935bc62723a212090baa9f98ada206fc14` | Simulator for ranking arrival/load/cache scenarios before expensive serving runs. Vidur is not final evidence. |
| Replication runtime | `https://github.com/vllm-project/vllm.git` | `fb5ec0dc9edfdba54882575e33e2a4215bd295b9` | Secondary runtime for repeating decisive comparisons when feasible. |

## Current Integration Boundary

- No SGLang, Vidur, or vLLM source is vendored here.
- The current harness talks to serving runtimes through an OpenAI-compatible backend and optional Prometheus scrape.
- The first E001 implementation pass should use an external router or minimal runtime plugin. If a fork becomes necessary, record the fork branch, changed upstream files, and patch list here before running canonical evidence.

## Refresh Rule

Changing any upstream pin invalidates canonical E001 comparability unless the experiment document is updated before the run. Historical results should cite the exact pin used by their manifest.
