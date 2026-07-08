import json

from adapter_cache_bench.analysis.report import generate_report


def _write_summary(tmp_path, run_id: str, **overrides):
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    summary = {
        "run_id": run_id,
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
        "slo_attainment_rate": 1,
        "mean_quality": 0.9,
        "quality_adjusted_goodput": 0.9,
        "quality_adjusted_goodput_per_memory_token": 0.09,
        "cache_hit_rate": 0.5,
        "cached_prompt_token_ratio": 0.5,
        "fragmentation_index": 1.0,
        "memory_token_footprint": 10,
        "eviction_count": 0,
        "evicted_tokens": 0,
        "adapter_distribution": {"qa": 1},
    }
    summary.update(overrides)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_report_generation_from_summary(tmp_path):
    _write_summary(tmp_path, "run")
    report = generate_report(
        tmp_path,
        tmp_path / "report.md",
        tmp_path / "tables",
        tmp_path / "figures",
    )
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "When is specialization worth its cache footprint?" in text
    assert "### Claim boundary" in text
    assert "Simulator regime map" in text
    assert "### Decision rule" in text
    assert "### Interpretation" in text
    assert "Generated table artifact paths:" in text
    assert "model or adapter specialization" in text
    assert (tmp_path / "tables" / "summaries.csv").exists()
    assert (tmp_path / "tables" / "claim_evidence.csv").exists()


def test_claim_boundary_scopes_real_server_rows_to_regime_workloads(tmp_path):
    _write_summary(
        tmp_path,
        "mock-regime",
        backend_kind="mock",
        backend_model="mock-causal-transformer",
        workload="regime_zipfian",
        cache_condition="warm",
    )
    _write_summary(
        tmp_path,
        "unrelated-vllm",
        backend_kind="vllm",
        backend_model="Qwen/Qwen2.5-1.5B-Instruct",
        workload="jsonl_eval",
        server_prefix_cache_queries=10,
        server_prefix_cache_hits=8,
    )

    report = generate_report(
        tmp_path,
        tmp_path / "report.md",
        tmp_path / "tables",
        tmp_path / "figures",
    )

    text = report.read_text(encoding="utf-8")
    assert (
        "| Real-server regime bridge | not supported here | "
        "no reset-isolated vLLM regime sweep in this artifact set |"
    ) in text
    assert (
        "| Prefix-cache causality | not established here | "
        "no positive server-side prefix/cache counters for `regime_*` vLLM bridge rows |"
    ) in text
    assert "Best comparable cache-footprint efficiency" in text


def test_report_scopes_unclassified_rows_and_nan_values(tmp_path):
    _write_summary(
        tmp_path,
        "legacy",
        backend_kind=None,
        backend_model=None,
        eviction_count=float("nan"),
    )

    report = generate_report(
        tmp_path,
        tmp_path / "report.md",
        tmp_path / "tables",
        tmp_path / "figures",
    )

    text = report.read_text(encoding="utf-8")
    assert "`legacy/unclassified` / `provenance unavailable`" in text
    assert "not as claim-supporting evidence" in text
    assert "| standard_lora | warm | specialist-adapter |" in text
    assert " nan " not in text


def test_report_skips_efficiency_winner_when_only_prefix_disabled_rows(tmp_path):
    _write_summary(
        tmp_path,
        "prefix-disabled",
        cache_condition="prefix_disabled",
        memory_token_footprint=0,
        quality_adjusted_goodput_per_memory_token=7.9,
    )

    report = generate_report(
        tmp_path,
        tmp_path / "report.md",
        tmp_path / "tables",
        tmp_path / "figures",
    )

    text = report.read_text(encoding="utf-8")
    assert "No comparable cache-footprint efficiency row is available" in text
    assert "Best comparable cache-footprint efficiency" not in text
