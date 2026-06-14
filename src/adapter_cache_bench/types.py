from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RequestRecord(BaseModel):
    request_id: str
    session_id: str
    tenant_id: str
    trust_group_id: str
    task_type: str
    prompt: str
    shared_prefix_id: str | None = None
    expected_adapter: str
    ground_truth: dict[str, Any] | str | None = None
    max_tokens: int = 64
    prompt_layout: str = "instruction_before_document"
    requires_json: bool = False


class RoutingDecision(BaseModel):
    request_id: str
    adapter_id: str
    policy_name: str
    score: float = 0.0
    reason: str = ""
    estimated_cached_prefix_tokens: int = 0


class RequestMetrics(BaseModel):
    prompt_tokens: int
    cached_prompt_tokens: int
    uncached_prompt_tokens: int
    prefill_ms: float
    decode_ms: float
    queue_ms: float
    ttft_ms: float
    itl_ms: float
    tpot_ms: float
    e2e_ms: float
    output_tokens: int


class QualityResult(BaseModel):
    task_type: str
    adapter_id: str
    score: float
    valid_json_rate: float | None = None
    schema_match: float | None = None
    field_f1: float | None = None
    exact_match_like_score: float | None = None
    unit_test_like_score: float | None = None
    rubric_score: float | None = None


class BackendResponse(BaseModel):
    request_id: str
    adapter_id: str
    text: str
    metrics: RequestMetrics
    quality: QualityResult


class BenchmarkSummary(BaseModel):
    run_id: str
    request_count: int
    backend_kind: str = "unknown"
    backend_model: str = "unknown"
    router_policy: str
    cache_model: str
    workload: str
    mean_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    mean_e2e_ms: float
    p50_e2e_ms: float
    p95_e2e_ms: float
    p99_e2e_ms: float
    mean_itl_ms: float
    mean_tpot_ms: float
    request_throughput: float
    token_throughput: float
    goodput_under_slo: float
    slo_attainment_rate: float
    mean_quality: float
    quality_adjusted_goodput: float
    quality_adjusted_goodput_per_memory_token: float
    cache_hit_rate: float
    cached_prompt_token_ratio: float
    fragmentation_index: float
    memory_token_footprint: int
    eviction_count: int = 0
    evicted_tokens: int = 0
    adapter_distribution: dict[str, int] = Field(default_factory=dict)
