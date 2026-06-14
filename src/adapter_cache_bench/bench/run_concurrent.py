from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from adapter_cache_bench.backends.base import make_backend
from adapter_cache_bench.backends.vllm_backend import VLLMBackend
from adapter_cache_bench.bench.metrics import summarize
from adapter_cache_bench.bench.run_workload import (
    build_manifest,
    scrape_backend_metrics,
)
from adapter_cache_bench.cache.cache_models import make_cache_model
from adapter_cache_bench.config import BenchmarkConfig, dump_config, load_config
from adapter_cache_bench.routing.base import make_router
from adapter_cache_bench.types import BackendResponse, RequestRecord, RoutingDecision
from adapter_cache_bench.workloads.generator import generate_workload


async def _generate_one(
    backend,
    request: RequestRecord,
    decision: RoutingDecision,
    cache_model,
) -> BackendResponse:
    if isinstance(backend, VLLMBackend):
        return await backend.async_generate(request, decision, cache_model)
    return await asyncio.to_thread(backend.generate, request, decision, cache_model)


async def _run_async(
    config: BenchmarkConfig,
    run_id: str | None = None,
    report_path: str | Path = "reports/adapter-cache-bench.md",
    tables_dir: str | Path = "reports/tables",
    generate_report_artifacts: bool = True,
) -> Path:
    run_id = run_id or f"{config.run_name}-{int(time.time() * 1000)}"
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_model = make_cache_model(config.cache)
    router = make_router(config.router)
    backend = make_backend(config.backend)
    requests = generate_workload(config.workload, config.cache)
    routed = [
        (request, router.route(request, config.adapters.adapter_ids, cache_model))
        for request in requests
    ]

    artifact_files = ["requests.jsonl", "summary.json", "config_resolved.yaml", "manifest.json"]
    before_metrics = scrape_backend_metrics(config, run_dir, "before")
    if before_metrics:
        artifact_files.append(before_metrics)

    semaphore = asyncio.Semaphore(max(1, config.backend.max_concurrency))

    async def worker(index: int, request: RequestRecord, decision: RoutingDecision):
        if config.backend.request_spacing_ms > 0:
            await asyncio.sleep(index * config.backend.request_spacing_ms / 1000.0)
        async with semaphore:
            return await _generate_one(backend, request, decision, cache_model)

    started = time.perf_counter()
    responses = await asyncio.gather(
        *(worker(index, request, decision) for index, (request, decision) in enumerate(routed))
    )
    wall_duration_s = max(0.001, time.perf_counter() - started)

    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for (request, decision), response in zip(routed, responses, strict=True):
            row = {
                "request": request.model_dump(mode="json"),
                "routing": decision.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
                "load": {
                    "max_concurrency": config.backend.max_concurrency,
                    "request_spacing_ms": config.backend.request_spacing_ms,
                },
            }
            handle.write(json.dumps(row) + "\n")

    after_metrics = scrape_backend_metrics(config, run_dir, "after")
    if after_metrics:
        artifact_files.append(after_metrics)

    summary = summarize(run_id, config, responses, cache_model, duration_s=wall_duration_s)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.model_dump(mode="json"), handle, indent=2)
    dump_config(config, run_dir / "config_resolved.yaml")
    manifest = build_manifest(
        run_id,
        config,
        request_count=len(responses),
        artifact_files=artifact_files,
    )
    manifest["max_concurrency"] = config.backend.max_concurrency
    manifest["request_spacing_ms"] = config.backend.request_spacing_ms
    manifest["wall_duration_s"] = wall_duration_s
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if generate_report_artifacts:
        from adapter_cache_bench.analysis.report import generate_report

        generate_report(config.output_dir, report_path=report_path, tables_dir=tables_dir)
    return run_dir


def run_concurrent(
    config: BenchmarkConfig,
    run_id: str | None = None,
    report_path: str | Path = "reports/adapter-cache-bench.md",
    tables_dir: str | Path = "reports/tables",
    generate_report_artifacts: bool = True,
) -> Path:
    return asyncio.run(
        _run_async(
            config,
            run_id=run_id,
            report_path=report_path,
            tables_dir=tables_dir,
            generate_report_artifacts=generate_report_artifacts,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    args = parser.parse_args()
    run_dir = run_concurrent(load_config(args.config))
    print(run_dir)


if __name__ == "__main__":
    main()
