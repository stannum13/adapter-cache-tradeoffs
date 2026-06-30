from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapter_cache_bench.config import BenchmarkConfig, dump_config


def git_metadata(cwd: str | Path = ".") -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status)}


def _optional_backend_field(config: BenchmarkConfig, name: str) -> Any | None:
    value = getattr(config.backend, name, None)
    if value is not None:
        return value
    return config.backend.extra_body.get(name)


_CLOUD_ENV_FIELDS = {
    "provider": "ACB_CLOUD_PROVIDER",
    "project": "ACB_CLOUD_PROJECT",
    "zone": "ACB_CLOUD_ZONE",
    "instance": "ACB_CLOUD_INSTANCE",
    "machine_type": "ACB_CLOUD_MACHINE_TYPE",
    "gpu_type": "ACB_CLOUD_GPU_TYPE",
    "gpu_count": "ACB_CLOUD_GPU_COUNT",
    "provisioning_model": "ACB_CLOUD_PROVISIONING_MODEL",
    "image": "ACB_CLOUD_IMAGE",
    "boot_disk_size": "ACB_CLOUD_BOOT_DISK_SIZE",
    "ttl_hours": "ACB_CLOUD_TTL_HOURS",
    "hourly_rate_usd": "ACB_CLOUD_HOURLY_RATE_USD",
}


def cloud_provenance_from_env() -> dict[str, Any]:
    provenance = {
        key: os.environ[env_name]
        for key, env_name in _CLOUD_ENV_FIELDS.items()
        if os.environ.get(env_name)
    }
    docker_image = os.environ.get("ACB_VLLM_IMAGE")
    if docker_image:
        provenance["vllm_image"] = docker_image
    return provenance


def build_manifest(
    run_id: str,
    config: BenchmarkConfig,
    request_count: int,
    artifact_files: list[str],
    created_at_unix_ms: int | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "run_name": config.run_name,
        "created_at_unix_ms": created_at_unix_ms or int(time.time() * 1000),
        "model": config.model.name,
        "backend": config.backend.kind,
        "backend_model": config.backend.model,
        "base_url": config.backend.base_url,
        "stream": config.backend.stream,
        "workload": config.workload.name,
        "router_policy": config.router.policy,
        "cache_model": config.cache.model,
        "cache_condition": config.cache.condition,
        "adapter_ids": config.adapters.adapter_ids,
        "adapter_model_names": config.backend.adapter_model_names,
        "max_concurrency": config.backend.max_concurrency,
        "request_spacing_ms": config.backend.request_spacing_ms,
        "request_count": request_count,
        "artifact_files": artifact_files,
        "metrics_scraped": config.backend.scrape_metrics,
        **git_metadata(),
    }
    request_timeout_s = _optional_backend_field(config, "request_timeout_s")
    if request_timeout_s is not None:
        manifest["request_timeout_s"] = request_timeout_s
    cloud = cloud_provenance_from_env()
    if cloud:
        manifest["cloud"] = cloud
    return manifest


def error_record(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


@dataclass
class RunState:
    run_dir: Path
    run_id: str
    config: BenchmarkConfig
    planned_request_count: int
    artifact_files: list[str] = field(
        default_factory=lambda: [
            "requests.jsonl",
            "summary.json",
            "config_resolved.yaml",
            "manifest.json",
            "status.json",
        ]
    )
    completed_request_count: int = 0
    failed_request_count: int = 0
    started_at_unix_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    _started_perf: float = field(default_factory=time.perf_counter)

    def initialize(self) -> None:
        dump_config(self.config, self.run_dir / "config_resolved.yaml")
        self.write_manifest()
        self.write_status("running")

    def add_artifact(self, name: str | None) -> None:
        if name and name not in self.artifact_files:
            self.artifact_files.append(name)
            self.write_manifest()

    def elapsed_s(self) -> float:
        return max(0.0, time.perf_counter() - self._started_perf)

    def write_manifest(self) -> None:
        manifest = build_manifest(
            self.run_id,
            self.config,
            request_count=self.planned_request_count,
            artifact_files=self.artifact_files,
            created_at_unix_ms=self.started_at_unix_ms,
        )
        with (self.run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    def write_status(self, status: str, exc: BaseException | None = None) -> None:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "status": status,
            "started_at_unix_ms": self.started_at_unix_ms,
            "updated_at_unix_ms": int(time.time() * 1000),
            "elapsed_s": self.elapsed_s(),
            "planned_request_count": self.planned_request_count,
            "completed_request_count": self.completed_request_count,
            "failed_request_count": self.failed_request_count,
        }
        if exc is not None:
            payload["exception_type"] = type(exc).__name__
            payload["exception_message"] = str(exc)
        with (self.run_dir / "status.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
