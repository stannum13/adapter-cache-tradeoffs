import json

import pandas as pd

from adapter_cache_bench.analysis.regime_science import (
    build_regime_policy_failure_matrix,
    write_regime_policy_failure_map,
)
from adapter_cache_bench.analysis.report import generate_report


def _write_run(
    runs_dir,
    run_id,
    *,
    workload,
    router_policy,
    cache_model,
    qag,
    strategy=None,
):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    summary = {
        "run_id": run_id,
        "request_count": 20,
        "router_policy": router_policy,
        "cache_model": cache_model,
        "workload": workload,
        "mean_ttft_ms": 10,
        "p50_ttft_ms": 10,
        "p95_ttft_ms": 10,
        "p99_ttft_ms": 10,
        "mean_e2e_ms": 20,
        "p50_e2e_ms": 20,
        "p95_e2e_ms": 20,
        "p99_e2e_ms": 20,
        "mean_itl_ms": 1,
        "mean_tpot_ms": 2,
        "request_throughput": 2,
        "token_throughput": 10,
        "goodput_under_slo": 2,
        "slo_attainment_rate": 1,
        "mean_quality": 0.9,
        "quality_adjusted_goodput": qag,
        "quality_adjusted_goodput_per_memory_token": qag / 100,
        "cache_hit_rate": 0.5,
        "cached_prompt_token_ratio": 0.5,
        "fragmentation_index": 1.0,
        "memory_token_footprint": 100,
        "eviction_count": 0,
        "evicted_tokens": 0,
        "adapter_distribution": {"qa": 1},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sweep_dimensions": {
                    "router": router_policy,
                    "cache": cache_model,
                    "workload": workload,
                    "seed": 17,
                    **({"strategy": strategy} if strategy is not None else {}),
                }
            }
        ),
        encoding="utf-8",
    )


def test_regime_policy_failure_matrix_uses_regime_rows_only():
    table = pd.DataFrame(
        [
            {
                "workload": "regime_uniform",
                "router_policy": "oracle",
                "cache_model": "activated_lora",
                "strategy": "oracle",
                "relative_regret": 0.0,
            },
            {
                "workload": "regime_uniform",
                "router_policy": "semantic",
                "cache_model": "standard_lora",
                "strategy": "standard_lora",
                "relative_regret": 0.25,
            },
            {
                "workload": "shared_doc_qa",
                "router_policy": "semantic",
                "cache_model": "standard_lora",
                "strategy": "standard_lora",
                "relative_regret": 0.4,
            },
            {
                "workload": "regime_zipfian",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "strategy": "activated_lora",
                "relative_regret": 0.1,
            },
        ]
    )

    matrix = build_regime_policy_failure_matrix(table)

    assert list(matrix.index) == ["regime_uniform", "regime_zipfian"]
    assert "shared_doc_qa" not in matrix.index
    assert "oracle" in matrix.columns
    assert "semantic / standard_lora" in matrix.columns
    assert matrix.loc["regime_uniform", "semantic / standard_lora"] == 0.25


def test_regime_policy_failure_matrix_keeps_cache_conditions_separate():
    table = pd.DataFrame(
        [
            {
                "workload": "regime_uniform",
                "router_policy": "semantic",
                "cache_model": "standard_lora",
                "strategy": "standard_lora",
                "sweep_cache_condition": "warm",
                "relative_regret": 0.1,
            },
            {
                "workload": "regime_uniform",
                "router_policy": "semantic",
                "cache_model": "standard_lora",
                "strategy": "standard_lora",
                "sweep_cache_condition": "prefix_disabled",
                "relative_regret": 0.4,
            },
        ]
    )

    matrix = build_regime_policy_failure_matrix(table)

    assert matrix.loc["regime_uniform", "semantic / standard_lora"] == 0.1
    assert matrix.loc["regime_uniform / prefix_disabled", "semantic / standard_lora"] == 0.4


def test_regime_policy_failure_map_writes_png(tmp_path):
    _write_run(
        tmp_path,
        "uniform-oracle",
        workload="regime_uniform",
        router_policy="oracle",
        cache_model="activated_lora",
        strategy="oracle",
        qag=10.0,
    )
    _write_run(
        tmp_path,
        "uniform-semantic",
        workload="regime_uniform",
        router_policy="semantic",
        cache_model="standard_lora",
        qag=7.5,
    )
    _write_run(
        tmp_path,
        "zipfian-cache-aware",
        workload="regime_zipfian",
        router_policy="cache_aware",
        cache_model="activated_lora",
        qag=8.0,
    )

    output = tmp_path / "figures" / "regime_policy_failure_map.png"
    path = write_regime_policy_failure_map(tmp_path, output)

    assert path == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_regime_policy_failure_map_skips_non_regime_inputs(tmp_path):
    _write_run(
        tmp_path,
        "shared-doc",
        workload="shared_doc_qa",
        router_policy="semantic",
        cache_model="standard_lora",
        qag=7.5,
    )

    path = write_regime_policy_failure_map(tmp_path, tmp_path / "plot.png")

    assert path is None
    assert not (tmp_path / "plot.png").exists()


def test_report_lists_regime_policy_failure_map_for_regime_runs(tmp_path):
    _write_run(
        tmp_path,
        "uniform-oracle",
        workload="regime_uniform",
        router_policy="oracle",
        cache_model="activated_lora",
        strategy="oracle",
        qag=10.0,
    )
    _write_run(
        tmp_path,
        "uniform-semantic",
        workload="regime_uniform",
        router_policy="semantic",
        cache_model="standard_lora",
        qag=7.5,
    )

    figures_dir = tmp_path / "figures"
    report = generate_report(tmp_path, tmp_path / "report.md", tmp_path / "tables", figures_dir)

    text = report.read_text(encoding="utf-8")
    assert "regime_policy_failure_map.png" in text
    assert (figures_dir / "regime_policy_failure_map.png").exists()


def test_report_skips_regime_policy_failure_map_for_non_regime_runs(tmp_path):
    _write_run(
        tmp_path,
        "shared-doc",
        workload="shared_doc_qa",
        router_policy="semantic",
        cache_model="standard_lora",
        qag=7.5,
    )

    figures_dir = tmp_path / "figures"
    report = generate_report(tmp_path, tmp_path / "report.md", tmp_path / "tables", figures_dir)

    text = report.read_text(encoding="utf-8")
    assert "regime_policy_failure_map.png" not in text
    assert not (figures_dir / "regime_policy_failure_map.png").exists()
