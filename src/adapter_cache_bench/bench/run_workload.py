from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import adapter_cache_bench.bench.run_state as run_state
from adapter_cache_bench.backends.base import make_backend
from adapter_cache_bench.backends.metrics_client import MetricsClient, prometheus_delta
from adapter_cache_bench.backends.server_control import prepare_backend_server
from adapter_cache_bench.cache.cache_models import make_cache_model
from adapter_cache_bench.config import BenchmarkConfig, load_config
from adapter_cache_bench.routing.base import make_router
from adapter_cache_bench.workloads.generator import generate_workload


def git_metadata(cwd: str | Path = ".") -> dict[str, Any]:
    return run_state.git_metadata(cwd)


def build_manifest(
    run_id: str,
    config: BenchmarkConfig,
    request_count: int,
    artifact_files: list[str],
) -> dict[str, Any]:
    return run_state.build_manifest(run_id, config, request_count, artifact_files)


def scrape_backend_metrics(config: BenchmarkConfig, run_dir: Path, label: str) -> str | None:
    if not config.backend.scrape_metrics:
        return None
    metrics_path = run_dir / f"backend_metrics_{label}.prom"
    try:
        metrics_text = MetricsClient(config.backend.metrics_url).scrape()
    except Exception as exc:  # pragma: no cover - network failures are environment-specific.
        (run_dir / f"backend_metrics_{label}_error.txt").write_text(str(exc), encoding="utf-8")
        return f"backend_metrics_{label}_error.txt"
    metrics_path.write_text(metrics_text, encoding="utf-8")
    return metrics_path.name


def backend_metrics_delta(run_dir: Path) -> dict[str, float]:
    before_path = run_dir / "backend_metrics_before.prom"
    after_path = run_dir / "backend_metrics_after.prom"
    if not before_path.exists() or not after_path.exists():
        return {}
    return prometheus_delta(
        before_path.read_text(encoding="utf-8"),
        after_path.read_text(encoding="utf-8"),
    )


def run(
    config: BenchmarkConfig,
    run_id: str | None = None,
    report_path: str | Path = "reports/adapter-cache-tradeoffs.md",
    tables_dir: str | Path = "reports/tables",
    generate_report_artifacts: bool = True,
) -> Path:
    run_id = run_id or f"{config.run_name}-{int(time.time() * 1000)}"
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    state = run_state.RunState(
        run_dir,
        run_id,
        config,
        planned_request_count=config.workload.request_count,
    )
    state.initialize()
    responses = []
    run_error: BaseException | None = None
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
        before_metrics = scrape_backend_metrics(config, run_dir, "before")
        state.add_artifact(before_metrics)

        with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
            for request in requests:
                decision = router.route(request, config.adapters.adapter_ids, cache_model)
                try:
                    response = backend.generate(request, decision, cache_model)
                except Exception as exc:
                    state.failed_request_count += 1
                    row = {
                        "request": request.model_dump(mode="json"),
                        "routing": decision.model_dump(mode="json"),
                        "error": run_state.error_record(exc),
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    raise
                responses.append(response)
                state.completed_request_count += 1
                row = {
                    "request": request.model_dump(mode="json"),
                    "routing": decision.model_dump(mode="json"),
                    "response": response.model_dump(mode="json"),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
        after_metrics = scrape_backend_metrics(config, run_dir, "after")
        state.add_artifact(after_metrics)

        from adapter_cache_bench.bench.metrics import summarize

        summary = summarize(
            run_id,
            config,
            responses,
            cache_model,
            backend_metrics=backend_metrics_delta(run_dir),
        )
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary.model_dump(mode="json"), handle, indent=2)

        if generate_report_artifacts:
            from adapter_cache_bench.analysis.report import generate_report

            generate_report(config.output_dir, report_path=report_path, tables_dir=tables_dir)
    except Exception as exc:
        run_error = exc
        raise
    finally:
        state.write_status("failed" if run_error else "complete", run_error)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    args = parser.parse_args()
    run_dir = run(load_config(args.config))
    print(run_dir)


if __name__ == "__main__":
    main()
