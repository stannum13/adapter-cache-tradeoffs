import json

from specialization_cache_frontier.analysis.report import generate_report


def test_report_generation_from_summary(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "run_id": "run",
        "request_count": 1,
        "router_policy": "semantic",
        "cache_model": "standard_lora",
        "workload": "shared_doc_qa",
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
        "request_throughput": 1,
        "token_throughput": 10,
        "goodput_under_slo": 1,
        "mean_quality": 0.9,
        "quality_adjusted_goodput": 0.9,
        "cache_hit_rate": 0.5,
        "cached_prompt_token_ratio": 0.5,
        "fragmentation_index": 1.0,
        "memory_token_footprint": 10,
        "adapter_distribution": {"qa": 1},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = generate_report(tmp_path, tmp_path / "report.md")
    assert report.exists()
    assert "When is specialization worth its cache footprint?" in report.read_text(encoding="utf-8")
