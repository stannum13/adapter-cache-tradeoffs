import json

import pandas as pd

from adapter_cache_bench.analysis.pareto import (
    pareto_frontier,
    workload_pareto_frontiers,
    write_pareto_frontier,
)


def test_pareto_frontier_keeps_nondominated_quality_latency_points():
    df = pd.DataFrame(
        [
            {"name": "fast-low", "mean_quality": 0.7, "p95_ttft_ms": 10.0},
            {"name": "slow-high", "mean_quality": 0.9, "p95_ttft_ms": 20.0},
            {"name": "slow-low", "mean_quality": 0.6, "p95_ttft_ms": 30.0},
        ]
    )

    frontier = pareto_frontier(df)

    assert frontier["name"].tolist() == ["fast-low", "slow-high"]


def test_workload_pareto_frontiers_group_by_workload():
    df = pd.DataFrame(
        [
            {
                "workload": "a",
                "router_policy": "semantic",
                "cache_model": "standard_lora",
                "mean_quality": 0.7,
                "p95_ttft_ms": 10.0,
            },
            {
                "workload": "b",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "mean_quality": 0.9,
                "p95_ttft_ms": 20.0,
            },
        ]
    )

    frontier = workload_pareto_frontiers(df)

    assert set(frontier["pareto_workload"]) == {"a", "b"}


def test_write_pareto_frontier_exports_csv(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "run_id": "run",
        "request_count": 1,
        "router_policy": "semantic",
        "cache_model": "standard_lora",
        "workload": "shared_doc_qa",
        "mean_quality": 0.9,
        "p95_ttft_ms": 10,
        "quality_adjusted_goodput": 1.0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    output = write_pareto_frontier(tmp_path, tmp_path / "pareto.csv")

    assert output.exists()
    assert "shared_doc_qa" in output.read_text(encoding="utf-8")
