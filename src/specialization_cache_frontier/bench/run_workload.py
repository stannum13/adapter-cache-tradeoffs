from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from specialization_cache_frontier.backends.mock_backend import MockBackend
from specialization_cache_frontier.cache.cache_models import make_cache_model
from specialization_cache_frontier.config import BenchmarkConfig, dump_config, load_config
from specialization_cache_frontier.routing.base import make_router
from specialization_cache_frontier.workloads.generator import generate_workload


def run(
    config: BenchmarkConfig,
    run_id: str | None = None,
    report_path: str | Path = "reports/specialization-cache-frontier.md",
) -> Path:
    run_id = run_id or f"{config.run_name}-{int(time.time() * 1000)}"
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_model = make_cache_model(config.cache)
    router = make_router(config.router)
    backend = MockBackend(config.backend)
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

    from specialization_cache_frontier.bench.metrics import summarize

    summary = summarize(run_id, config, responses, cache_model)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.model_dump(mode="json"), handle, indent=2)
    dump_config(config, run_dir / "config_resolved.yaml")

    from specialization_cache_frontier.analysis.report import generate_report

    generate_report(config.output_dir, report_path=report_path)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_dir = run(load_config(args.config))
    print(run_dir)


if __name__ == "__main__":
    main()
