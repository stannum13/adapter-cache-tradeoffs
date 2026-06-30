from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from adapter_cache_bench.backends.base import make_backend
from adapter_cache_bench.backends.server_control import prepare_backend_server
from adapter_cache_bench.backends.vllm_backend import VLLMBackend
from adapter_cache_bench.bench.metrics import summarize
from adapter_cache_bench.bench.run_state import RunState, error_record
from adapter_cache_bench.bench.run_workload import (
    backend_metrics_delta,
    scrape_backend_metrics,
)
from adapter_cache_bench.cache.cache_models import make_cache_model
from adapter_cache_bench.config import BenchmarkConfig, load_config
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
    report_path: str | Path = "reports/adapter-cache-tradeoffs.md",
    tables_dir: str | Path = "reports/tables",
    generate_report_artifacts: bool = True,
) -> Path:
    run_id = run_id or f"{config.run_name}-{int(time.time() * 1000)}"
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    state = RunState(
        run_dir,
        run_id,
        config,
        planned_request_count=config.workload.request_count,
    )
    state.initialize()
    responses: list[BackendResponse] = []
    request_errors: list[BaseException] = []
    run_error: BaseException | None = None
    cache_model = None
    wall_duration_s = 0.0
    try:
        requests = generate_workload(config.workload, config.cache)
        state.planned_request_count = len(requests)
        state.write_manifest()
        state.write_status("running")
        reset_artifact = prepare_backend_server(config.backend, run_dir)
        state.add_artifact(reset_artifact)
        cache_model = make_cache_model(config.cache)
        router = make_router(config.router)
        backend = make_backend(config.backend)
        routed = [
            (request, router.route(request, config.adapters.adapter_ids, cache_model))
            for request in requests
        ]

        before_metrics = scrape_backend_metrics(config, run_dir, "before")
        state.add_artifact(before_metrics)

        semaphore = asyncio.Semaphore(max(1, config.backend.max_concurrency))
        load = {
            "max_concurrency": config.backend.max_concurrency,
            "request_spacing_ms": config.backend.request_spacing_ms,
        }

        async def worker(
            index: int,
            request: RequestRecord,
            decision: RoutingDecision,
        ) -> tuple[RequestRecord, RoutingDecision, BackendResponse | None, BaseException | None]:
            if config.backend.request_spacing_ms > 0:
                await asyncio.sleep(index * config.backend.request_spacing_ms / 1000.0)
            async with semaphore:
                try:
                    response = await _generate_one(backend, request, decision, cache_model)
                except Exception as exc:
                    return request, decision, None, exc
                return request, decision, response, None

        tasks = [
            asyncio.create_task(worker(index, request, decision))
            for index, (request, decision) in enumerate(routed)
        ]
        started = time.perf_counter()
        with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
            for task in asyncio.as_completed(tasks):
                request, decision, response, exc = await task
                row = {
                    "request": request.model_dump(mode="json"),
                    "routing": decision.model_dump(mode="json"),
                    "load": load,
                }
                if exc is None and response is not None:
                    responses.append(response)
                    state.completed_request_count += 1
                    row["response"] = response.model_dump(mode="json")
                else:
                    state.failed_request_count += 1
                    request_errors.append(exc or RuntimeError("request failed without exception"))
                    row["error"] = error_record(request_errors[-1])
                handle.write(json.dumps(row) + "\n")
                handle.flush()
        wall_duration_s = max(0.001, time.perf_counter() - started)

        after_metrics = scrape_backend_metrics(config, run_dir, "after")
        state.add_artifact(after_metrics)

        summary = summarize(
            run_id,
            config,
            responses,
            cache_model,
            duration_s=wall_duration_s,
            backend_metrics=backend_metrics_delta(run_dir),
            requests=requests,
        )
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary.model_dump(mode="json"), handle, indent=2)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["wall_duration_s"] = wall_duration_s
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        if request_errors:
            raise RuntimeError(f"{len(request_errors)} concurrent request(s) failed")

        if generate_report_artifacts:
            from adapter_cache_bench.analysis.report import generate_report

            generate_report(config.output_dir, report_path=report_path, tables_dir=tables_dir)
    except Exception as exc:
        run_error = exc
        raise
    finally:
        if cache_model is None:
            wall_duration_s = max(wall_duration_s, state.elapsed_s())
        state.write_status("failed" if run_error else "complete", run_error)
    return run_dir


def run_concurrent(
    config: BenchmarkConfig,
    run_id: str | None = None,
    report_path: str | Path = "reports/adapter-cache-tradeoffs.md",
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
