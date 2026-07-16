import pytest

from adapter_cache_bench.bench.metrics import summarize
from adapter_cache_bench.cache.standard_lora_cache import StandardLoRACache
from adapter_cache_bench.config import BackendConfig, BenchmarkConfig, CacheConfig
from adapter_cache_bench.types import BackendResponse, QualityResult, RequestMetrics


def _response(request_id: str, ttft_ms: float, quality: float) -> BackendResponse:
    return BackendResponse(
        request_id=request_id,
        adapter_id="qa",
        text="answer",
        metrics=RequestMetrics(
            prompt_tokens=10,
            cached_prompt_tokens=0,
            uncached_prompt_tokens=10,
            prefill_ms=0.0,
            decode_ms=0.0,
            queue_ms=0.0,
            ttft_ms=ttft_ms,
            itl_ms=0.0,
            tpot_ms=0.0,
            e2e_ms=100.0,
            output_tokens=1,
        ),
        quality=QualityResult(task_type="qa", adapter_id="qa", score=quality),
    )


def test_quality_adjusted_goodput_sums_quality_only_for_slo_passing_requests():
    config = BenchmarkConfig(
        backend=BackendConfig(ttft_slo_ms=50.0),
        cache=CacheConfig(block_size=2),
    )
    cache = StandardLoRACache(config.cache)
    responses = [
        _response("fast-low-quality", ttft_ms=40.0, quality=0.25),
        _response("slow-high-quality", ttft_ms=60.0, quality=1.0),
    ]

    summary = summarize("unit", config, responses, cache, duration_s=0.5)

    assert summary.goodput_under_slo == pytest.approx(2.0)
    assert summary.mean_quality == pytest.approx(0.625)
    assert summary.quality_adjusted_goodput == pytest.approx(0.5)
