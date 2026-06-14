from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = "mock-causal-transformer"
    max_context_tokens: int = 4096


class AdapterConfig(BaseModel):
    adapter_ids: list[str] = Field(
        default_factory=lambda: ["qa", "json", "summary", "code", "multitask"]
    )
    default_adapter: str = "multitask"
    cold_start_penalty_ms: float = 25.0


class CacheConfig(BaseModel):
    model: str = "standard_lora"
    block_size: int = 16
    max_memory_tokens: int | None = None
    eviction_policy: str = "lru"
    invocation_markers: dict[str, str] = Field(
        default_factory=lambda: {
            "qa": "<ADAPTER:qa>",
            "json": "<ADAPTER:json>",
            "summary": "<ADAPTER:summary>",
            "code": "<ADAPTER:code>",
            "multitask": "<ADAPTER:multitask>",
        }
    )
    isolation_scope: str = "trust_group"
    cache_salt: str | None = None
    copy_on_write_delta_fraction: float = 0.15


class RouterConfig(BaseModel):
    policy: str = "cache_aware"
    seed: int = 7
    alpha: float = 0.01
    beta: float = 0.001
    gamma: float = 0.10
    delta: float = 0.25
    epsilon: float = 0.05


class WorkloadConfig(BaseModel):
    name: str = "shared_doc_qa"
    dataset_path: str | None = None
    request_count: int = 40
    shared_document_count: int = 2
    sessions: int = 8
    tenants: int = 2
    max_tokens: int = 64
    seed: int = 11
    document_tokens: int = 180


class BackendConfig(BaseModel):
    kind: str = "mock"
    seed: int = 13
    prefill_ms_per_token: float = 0.35
    decode_ms_per_token: float = 1.2
    first_token_ms: float = 8.0
    queue_ms_min: float = 0.0
    queue_ms_max: float = 6.0
    ttft_slo_ms: float = 250.0
    e2e_slo_ms: float = 1200.0
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "mock-causal-transformer"
    temperature: float = 0.0
    adapter_model_names: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    scrape_metrics: bool = False
    metrics_url: str = "http://localhost:8000/metrics"
    max_concurrency: int = 1
    request_spacing_ms: float = 0.0


class BenchmarkConfig(BaseModel):
    run_name: str = "small"
    model: ModelConfig = Field(default_factory=ModelConfig)
    adapters: AdapterConfig = Field(default_factory=AdapterConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    output_dir: str = "artifacts/runs"
    matrix: dict[str, list[str | int]] = Field(default_factory=dict)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | list[str | Path] | tuple[str | Path, ...]) -> BenchmarkConfig:
    paths: list[str | Path]
    if isinstance(path, str | Path):
        paths = [path]
    else:
        paths = list(path)
    merged: dict[str, Any] = {}
    for item in paths:
        merged = deep_merge(merged, load_yaml(item))
    return BenchmarkConfig.model_validate(merged)


def dump_config(config: BenchmarkConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
