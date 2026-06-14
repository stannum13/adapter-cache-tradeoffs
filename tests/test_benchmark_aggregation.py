import json

from adapter_cache_bench.bench.aggregate import (
    cache_model_means,
    load_request_rows,
    load_summaries,
    repeated_seed_summary,
    router_means,
    workload_leaders,
    write_analysis_tables,
)
from adapter_cache_bench.bench.compare import compare_runs
from adapter_cache_bench.bench.run_concurrent import run_concurrent
from adapter_cache_bench.bench.run_workload import run
from adapter_cache_bench.config import (
    BackendConfig,
    BenchmarkConfig,
    CacheConfig,
    RouterConfig,
    WorkloadConfig,
)


def test_benchmark_run_writes_artifacts(tmp_path):
    config = BenchmarkConfig(
        run_name="test",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(name="mixed_tasks_same_doc", request_count=8, document_tokens=32),
        cache=CacheConfig(model="activated_lora", block_size=4),
        router=RouterConfig(policy="cache_aware"),
    )
    run_dir = run(
        config,
        run_id="unit",
        report_path=tmp_path / "report.md",
        tables_dir=tmp_path / "tables",
        generate_report_artifacts=True,
    )
    assert (run_dir / "requests.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config_resolved.yaml").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "unit"
    assert manifest["request_count"] == 8
    assert manifest["cache_model"] == "activated_lora"
    assert "git_commit" in manifest
    assert "git_dirty" in manifest
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["slo_attainment_rate"] >= 0.0
    assert summary["quality_adjusted_goodput_per_memory_token"] >= 0.0


def test_benchmark_run_scrapes_backend_metrics_when_enabled(tmp_path, monkeypatch):
    from adapter_cache_bench.bench import run_workload

    class FakeMetricsClient:
        def __init__(self, metrics_url):
            self.metrics_url = metrics_url

        def scrape(self):
            return "vllm:num_requests_running 0\n"

    monkeypatch.setattr(run_workload, "MetricsClient", FakeMetricsClient)
    config = BenchmarkConfig(
        run_name="test",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(name="mixed_tasks_same_doc", request_count=2, document_tokens=24),
        cache=CacheConfig(model="activated_lora", block_size=4),
        router=RouterConfig(policy="cache_aware"),
        backend=BackendConfig(kind="mock", scrape_metrics=True, metrics_url="http://unit/metrics"),
    )

    run_dir = run(
        config,
        run_id="unit-metrics",
        report_path=tmp_path / "report.md",
        tables_dir=tmp_path / "tables",
        generate_report_artifacts=False,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics_scraped"] is True
    assert "backend_metrics_before.prom" in manifest["artifact_files"]
    assert "backend_metrics_after.prom" in manifest["artifact_files"]
    assert (run_dir / "backend_metrics_before.prom").read_text(encoding="utf-8").startswith("vllm")


def test_benchmark_run_can_skip_report_artifacts(tmp_path):
    config = BenchmarkConfig(
        run_name="test",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(name="mixed_tasks_same_doc", request_count=4, document_tokens=24),
        cache=CacheConfig(model="activated_lora", block_size=4),
        router=RouterConfig(policy="cache_aware"),
    )

    run(
        config,
        run_id="unit-no-report",
        report_path=tmp_path / "report.md",
        tables_dir=tmp_path / "tables",
        generate_report_artifacts=False,
    )

    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "tables").exists()


def test_concurrent_benchmark_run_writes_artifacts(tmp_path):
    config = BenchmarkConfig(
        run_name="concurrent-test",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(name="mixed_tasks_same_doc", request_count=6, document_tokens=24),
        cache=CacheConfig(model="activated_lora", block_size=4),
        router=RouterConfig(policy="cache_aware"),
        backend=BackendConfig(kind="mock", max_concurrency=3),
    )

    run_dir = run_concurrent(
        config,
        run_id="unit-concurrent",
        report_path=tmp_path / "report.md",
        tables_dir=tmp_path / "tables",
        generate_report_artifacts=False,
    )

    assert (run_dir / "requests.jsonl").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    first_row = json.loads((run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["request_count"] == 6
    assert summary["request_throughput"] > 0.0
    assert manifest["max_concurrency"] == 3
    assert first_row["load"]["max_concurrency"] == 3


def test_load_request_rows_reads_layout_and_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "router_policy": "semantic",
                "cache_model": "activated_lora",
                "workload": "prompt_layout_ablation",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "requests.jsonl").write_text(
        json.dumps(
            {
                "request": {
                    "request_id": "r1",
                    "prompt_layout": "document_before_instruction",
                    "task_type": "qa",
                },
                "routing": {"adapter_id": "qa"},
                "response": {
                    "metrics": {
                        "ttft_ms": 10.0,
                        "e2e_ms": 20.0,
                        "cached_prompt_tokens": 8,
                        "prompt_tokens": 16,
                    },
                    "quality": {"score": 0.9},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    df = load_request_rows(tmp_path)

    assert df.iloc[0]["prompt_layout"] == "document_before_instruction"
    assert df.iloc[0]["ttft_ms"] == 10.0


def test_load_summaries_reads_concurrency_metadata_from_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "workload": "jsonl_eval",
                "quality_adjusted_goodput": 1.0,
                "mean_quality": 0.8,
                "p95_ttft_ms": 100.0,
                "memory_token_footprint": 10,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"max_concurrency": 8, "request_spacing_ms": 5.0, "wall_duration_s": 12.5}),
        encoding="utf-8",
    )

    df = load_summaries(tmp_path)

    assert df.iloc[0]["max_concurrency"] == 8
    assert df.iloc[0]["request_spacing_ms"] == 5.0
    assert df.iloc[0]["wall_duration_s"] == 12.5


def test_analysis_tables_rank_workloads_and_export_csv(tmp_path):
    import pandas as pd

    summaries = pd.DataFrame(
        [
            {
                "workload": "shared_doc_qa",
                "router_policy": "semantic",
                "cache_model": "standard_lora",
                "quality_adjusted_goodput": 1.0,
                "mean_quality": 0.9,
                "p95_ttft_ms": 40.0,
                "cache_hit_rate": 0.5,
                "memory_token_footprint": 100,
                "fragmentation_index": 2.0,
                "eviction_count": 0,
                "evicted_tokens": 0,
            },
            {
                "workload": "shared_doc_qa",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "quality_adjusted_goodput": 1.4,
                "mean_quality": 0.88,
                "p95_ttft_ms": 20.0,
                "cache_hit_rate": 0.8,
                "memory_token_footprint": 70,
                "fragmentation_index": 1.1,
                "eviction_count": 0,
                "evicted_tokens": 0,
            },
        ]
    )
    requests = pd.DataFrame()

    leaders = workload_leaders(summaries)
    cache_means = cache_model_means(summaries)
    routers = router_means(summaries)
    paths = write_analysis_tables(summaries, requests, tmp_path / "tables")

    assert leaders.iloc[0]["router_policy"] == "cache_aware"
    assert cache_means.iloc[0]["cache_model"] == "activated_lora"
    assert routers.iloc[0]["router_policy"] == "cache_aware"
    assert paths["workload_leaders"].exists()


def test_repeated_seed_summary_reports_mean_and_std():
    import pandas as pd

    summaries = pd.DataFrame(
        [
            {
                "run_id": "a",
                "workload": "shared_doc_qa",
                "router_policy": "semantic",
                "cache_model": "standard_lora",
                "quality_adjusted_goodput": 1.0,
                "mean_quality": 0.8,
                "p95_ttft_ms": 10.0,
            },
            {
                "run_id": "b",
                "workload": "shared_doc_qa",
                "router_policy": "semantic",
                "cache_model": "standard_lora",
                "quality_adjusted_goodput": 3.0,
                "mean_quality": 1.0,
                "p95_ttft_ms": 20.0,
            },
        ]
    )

    summary = repeated_seed_summary(summaries)

    assert summary.iloc[0]["run_count"] == 2
    assert summary.iloc[0]["quality_adjusted_goodput_mean"] == 2.0
    assert summary.iloc[0]["quality_adjusted_goodput_std"] > 0


def test_compare_runs_returns_leader_tables(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "request_count": 1,
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
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
        ),
        encoding="utf-8",
    )

    tables = compare_runs(tmp_path)

    assert tables["workload_leaders"].iloc[0]["cache_model"] == "activated_lora"
