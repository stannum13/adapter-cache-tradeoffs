# Figures

These committed snapshots are generated from local benchmark artifacts with:

```bash
make report
make whitepaper-figure
make large-model-figures
```

Raw run artifacts remain ignored under `artifacts/runs/`. Regenerate figures
after new runs and copy publication-worthy plots here when they should appear on
GitHub.

Included snapshots:

- `whitepaper_specialization_cache_tradeoff.png`
- `whitepaper_specialization_cache_tradeoff.pdf`
- `large_model_overlap_confidence.png`
- `large_model_adapter_quality.png`
- `large_model_adapter_concurrent.png`
- `source_backed_qwen7b_adapter_seeds.png`
- `source_backed_qwen7b_expanded.png`
- `quality_vs_p95_ttft.png`
- `cache_hit_rate_by_policy_model.png`
- `quality_adjusted_goodput_by_router.png`
- `memory_token_footprint_by_cache.png`
- `prompt_layout_ablation.png`
- `adapter_strategy_frontier.png`
- `concurrency_p95_ttft.png`
- `concurrency_qag.png`
