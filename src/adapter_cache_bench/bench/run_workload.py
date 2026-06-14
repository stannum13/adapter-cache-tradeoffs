from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from adapter_cache_bench.backends.base import make_backend
from adapter_cache_bench.cache.cache_models import make_cache_model
from adapter_cache_bench.config import BenchmarkConfig, dump_config, load_config
from adapter_cache_bench.routing.base import make_router
from adapter_cache_bench.workloads.generator import generate_workload


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


def build_manifest(
    run_id: str,
    config: BenchmarkConfig,
    request_count: int,
    artifact_files: list[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_name": config.run_name,
        "created_at_unix_ms": int(time.time() * 1000),
        "model": config.model.name,
        "backend": config.backend.kind,
        "workload": config.workload.name,
        "router_policy": config.router.policy,
        "cache_model": config.cache.model,
        "adapter_ids": config.adapters.adapter_ids,
        "request_count": request_count,
        "artifact_files": artifact_files,
        **git_metadata(),
    }


def run(
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

    responses = []
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for request in requests:
            decision = router.route(request, config.adapters.adapter_ids, cache_model)
            response = backend.generate(request, decision, cache_model)
            responses.append(response)
            row = {
                "request": request.model_dump(mode="json"),
                "routing": decision.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
            }
            handle.write(json.dumps(row) + "\n")

    from adapter_cache_bench.bench.metrics import summarize

    summary = summarize(run_id, config, responses, cache_model)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.model_dump(mode="json"), handle, indent=2)
    dump_config(config, run_dir / "config_resolved.yaml")
    manifest = build_manifest(
        run_id,
        config,
        request_count=len(responses),
        artifact_files=["requests.jsonl", "summary.json", "config_resolved.yaml", "manifest.json"],
    )
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if generate_report_artifacts:
        from adapter_cache_bench.analysis.report import generate_report

        generate_report(config.output_dir, report_path=report_path, tables_dir=tables_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    args = parser.parse_args()
    run_dir = run(load_config(args.config))
    print(run_dir)


if __name__ == "__main__":
    main()
