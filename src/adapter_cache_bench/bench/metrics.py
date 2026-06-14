from __future__ import annotations

from collections import Counter

from adapter_cache_bench.cache.cache_models import CacheModel
from adapter_cache_bench.config import BenchmarkConfig
from adapter_cache_bench.types import BackendResponse, BenchmarkSummary


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[index]


def summarize(
    run_id: str, config: BenchmarkConfig, responses: list[BackendResponse], cache_model: CacheModel
) -> BenchmarkSummary:
    ttft = [response.metrics.ttft_ms for response in responses]
    e2e = [response.metrics.e2e_ms for response in responses]
    output_tokens = [response.metrics.output_tokens for response in responses]
    duration_s = max(0.001, sum(e2e) / 1000.0)
    good = [
        response for response in responses if response.metrics.ttft_ms <= config.backend.ttft_slo_ms
    ]
    mean_quality = sum(response.quality.score for response in responses) / max(1, len(responses))
    adapter_distribution = Counter(response.adapter_id for response in responses)
    goodput_under_slo = len(good) / duration_s
    quality_adjusted_goodput = goodput_under_slo * mean_quality
    memory_tokens = cache_model.memory_tokens()
    return BenchmarkSummary(
        run_id=run_id,
        request_count=len(responses),
        backend_kind=config.backend.kind,
        backend_model=config.backend.model,
        router_policy=config.router.policy,
        cache_model=config.cache.model,
        workload=config.workload.name,
        mean_ttft_ms=sum(ttft) / max(1, len(ttft)),
        p50_ttft_ms=percentile(ttft, 0.50),
        p95_ttft_ms=percentile(ttft, 0.95),
        p99_ttft_ms=percentile(ttft, 0.99),
        mean_e2e_ms=sum(e2e) / max(1, len(e2e)),
        p50_e2e_ms=percentile(e2e, 0.50),
        p95_e2e_ms=percentile(e2e, 0.95),
        p99_e2e_ms=percentile(e2e, 0.99),
        mean_itl_ms=sum(response.metrics.itl_ms for response in responses) / max(1, len(responses)),
        mean_tpot_ms=sum(response.metrics.tpot_ms for response in responses)
        / max(1, len(responses)),
        request_throughput=len(responses) / duration_s,
        token_throughput=sum(output_tokens) / duration_s,
        goodput_under_slo=goodput_under_slo,
        slo_attainment_rate=len(good) / max(1, len(responses)),
        mean_quality=mean_quality,
        quality_adjusted_goodput=quality_adjusted_goodput,
        quality_adjusted_goodput_per_memory_token=quality_adjusted_goodput
        / max(1, memory_tokens),
        cache_hit_rate=cache_model.cache_hit_rate(),
        cached_prompt_token_ratio=cache_model.cached_prompt_token_ratio(),
        fragmentation_index=cache_model.fragmentation_index(),
        memory_token_footprint=memory_tokens,
        eviction_count=cache_model.eviction_count(),
        evicted_tokens=cache_model.evicted_tokens(),
        adapter_distribution=dict(adapter_distribution),
    )
