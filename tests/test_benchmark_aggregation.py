import importlib
import json

import pytest

from adapter_cache_bench.analysis.adapter_cache_metrics import build_adapter_cache_metrics
from adapter_cache_bench.analysis.benchmark_v0 import benchmark_v0_summary
from adapter_cache_bench.backends.mock_backend import MockBackend
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
    assert (run_dir / "status.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "unit"
    assert manifest["request_count"] == 8
    assert manifest["cache_model"] == "activated_lora"
    assert manifest["backend_model"] == "mock-causal-transformer"
    assert manifest["base_url"] == "http://localhost:8000/v1"
    assert manifest["stream"] is False
    assert manifest["adapter_model_names"] == {}
    assert manifest["max_concurrency"] == 1
    assert manifest["request_spacing_ms"] == 0.0
    assert "status.json" in manifest["artifact_files"]
    assert "git_commit" in manifest
    assert "git_dirty" in manifest
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["completed_request_count"] == 8
    assert status["failed_request_count"] == 0
    assert status["elapsed_s"] >= 0.0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["slo_attainment_rate"] >= 0.0
    assert summary["quality_adjusted_goodput_per_memory_token"] >= 0.0


def test_benchmark_run_marks_failed_status_on_request_exception(tmp_path, monkeypatch):
    run_workload_module = importlib.import_module("adapter_cache_bench.bench.run_workload")

    class FlakyBackend:
        def __init__(self, config):
            self.inner = MockBackend(config)

        def generate(self, request, decision, cache_model):
            if request.request_id.endswith("00001"):
                raise RuntimeError("unit boom")
            return self.inner.generate(request, decision, cache_model)

    monkeypatch.setattr(
        run_workload_module,
        "make_backend",
        lambda backend_config: FlakyBackend(backend_config),
    )
    config = BenchmarkConfig(
        run_name="test-failed",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(name="mixed_tasks_same_doc", request_count=4, document_tokens=24),
        cache=CacheConfig(model="activated_lora", block_size=4),
        router=RouterConfig(policy="cache_aware"),
    )

    with pytest.raises(RuntimeError, match="unit boom"):
        run_workload_module.run(
            config,
            run_id="unit-failed",
            report_path=tmp_path / "report.md",
            tables_dir=tmp_path / "tables",
            generate_report_artifacts=False,
        )

    run_dir = tmp_path / "unit-failed"
    rows = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "manifest.json").exists()
    assert len(rows) == 2
    assert "response" in rows[0]
    assert rows[1]["error"]["type"] == "RuntimeError"
    assert rows[1]["error"]["message"] == "unit boom"
    assert status["status"] == "failed"
    assert status["completed_request_count"] == 1
    assert status["failed_request_count"] == 1
    assert status["exception_type"] == "RuntimeError"
    assert status["exception_message"] == "unit boom"


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
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["backend_metrics"]["vllm:num_requests_running"] == 0.0


def test_backend_metrics_delta_is_written_to_summary(tmp_path, monkeypatch):
    from adapter_cache_bench.bench import run_workload

    class FakeMetricsClient:
        calls = 0

        def __init__(self, metrics_url):
            self.metrics_url = metrics_url

        def scrape(self):
            FakeMetricsClient.calls += 1
            if FakeMetricsClient.calls == 1:
                return (
                    "# HELP vllm:prefix_cache_queries_total test\n"
                    'vllm:prefix_cache_queries_total{model="m"} 10\n'
                    'vllm:prefix_cache_hits_total{model="m"} 4\n'
                )
            return (
                'vllm:prefix_cache_queries_total{model="m"} 22\n'
                'vllm:prefix_cache_hits_total{model="m"} 13\n'
            )

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
        run_id="unit-metrics-delta",
        report_path=tmp_path / "report.md",
        tables_dir=tmp_path / "tables",
        generate_report_artifacts=False,
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["backend_metrics"]["vllm:prefix_cache_queries_total"] == 12.0
    assert summary["backend_metrics"]["vllm:prefix_cache_hits_total"] == 9.0


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


def test_concurrent_benchmark_streams_error_rows_and_failed_status(tmp_path, monkeypatch):
    run_concurrent_module = importlib.import_module("adapter_cache_bench.bench.run_concurrent")

    class FlakyBackend:
        def __init__(self, config):
            self.inner = MockBackend(config)

        def generate(self, request, decision, cache_model):
            if request.request_id.endswith("00001"):
                raise RuntimeError("concurrent unit boom")
            return self.inner.generate(request, decision, cache_model)

    monkeypatch.setattr(
        run_concurrent_module,
        "make_backend",
        lambda backend_config: FlakyBackend(backend_config),
    )
    config = BenchmarkConfig(
        run_name="concurrent-failed",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(name="mixed_tasks_same_doc", request_count=4, document_tokens=24),
        cache=CacheConfig(model="activated_lora", block_size=4),
        router=RouterConfig(policy="cache_aware"),
        backend=BackendConfig(kind="mock", max_concurrency=2),
    )

    with pytest.raises(RuntimeError, match=r"1 concurrent request\(s\) failed"):
        run_concurrent_module.run_concurrent(
            config,
            run_id="unit-concurrent-failed",
            report_path=tmp_path / "report.md",
            tables_dir=tmp_path / "tables",
            generate_report_artifacts=False,
        )

    run_dir = tmp_path / "unit-concurrent-failed"
    rows = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    error_rows = [row for row in rows if "error" in row]
    response_rows = [row for row in rows if "response" in row]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert len(response_rows) == 3
    assert len(error_rows) == 1
    assert error_rows[0]["error"]["type"] == "RuntimeError"
    assert error_rows[0]["error"]["message"] == "concurrent unit boom"
    assert error_rows[0]["load"]["max_concurrency"] == 2
    assert summary["request_count"] == 3
    assert status["status"] == "failed"
    assert status["completed_request_count"] == 3
    assert status["failed_request_count"] == 1
    assert status["exception_type"] == "RuntimeError"
    assert status["exception_message"] == "1 concurrent request(s) failed"
    assert manifest["wall_duration_s"] > 0.0


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
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "max_concurrency": 8,
                "sweep_dimensions": {"strategy": "specialists", "overlap_fraction": 0.75},
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
    assert df.iloc[0]["max_concurrency"] == 8
    assert df.iloc[0]["sweep_strategy"] == "specialists"
    assert df.iloc[0]["sweep_overlap_fraction"] == 0.75


def test_load_request_rows_skips_error_rows(tmp_path):
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
        "\n".join(
            [
                json.dumps(
                    {
                        "request": {
                            "request_id": "r1",
                            "prompt_layout": "document_before_instruction",
                            "task_type": "qa",
                        },
                        "routing": {"adapter_id": "qa"},
                        "error": {"type": "RuntimeError", "message": "boom"},
                    }
                ),
                json.dumps(
                    {
                        "request": {
                            "request_id": "r2",
                            "prompt_layout": "document_before_instruction",
                            "task_type": "qa",
                        },
                        "routing": {"adapter_id": "qa"},
                        "response": {
                            "metrics": {
                                "ttft_ms": 12.0,
                                "e2e_ms": 24.0,
                                "cached_prompt_tokens": 4,
                                "prompt_tokens": 16,
                            },
                            "quality": {"score": 0.7},
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    df = load_request_rows(tmp_path)

    assert list(df["request_id"]) == ["r2"]
    assert df.iloc[0]["ttft_ms"] == 12.0


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


def test_load_summaries_flattens_backend_metrics_and_sweep_dimensions(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "workload": "controlled_overlap",
                "quality_adjusted_goodput": 1.0,
                "mean_quality": 0.8,
                "p95_ttft_ms": 100.0,
                "memory_token_footprint": 10,
                "backend_metrics": {
                    "vllm:prefix_cache_queries_total": 10.0,
                    "vllm:prefix_cache_hits_total": 7.0,
                    "vllm:prompt_tokens_cached_total": 99.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sweep_dimensions": {
                    "strategy": "specialists",
                    "overlap_fraction": 0.75,
                    "adapter_count": 4,
                }
            }
        ),
        encoding="utf-8",
    )

    df = load_summaries(tmp_path)

    assert df.iloc[0]["backend_metric:vllm:prefix_cache_hits_total"] == 7.0
    assert df.iloc[0]["server_prefix_cache_queries"] == 10.0
    assert df.iloc[0]["server_prefix_cache_hits"] == 7.0
    assert df.iloc[0]["server_prefix_cache_hit_rate"] == 0.7
    assert df.iloc[0]["server_prompt_tokens_cached"] == 99.0
    assert df.iloc[0]["sweep_strategy"] == "specialists"
    assert df.iloc[0]["sweep_overlap_fraction"] == 0.75
    assert df.iloc[0]["sweep_adapter_count"] == 4


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
    assert paths["adapter_cache_metrics"].exists()


def test_adapter_cache_metrics_join_request_and_server_evidence(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "request_count": 2,
                "backend_kind": "vllm",
                "backend_model": "unit-model",
                "router_policy": "semantic",
                "cache_model": "activated_lora",
                "workload": "jsonl_eval",
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
                "backend_metrics": {
                    "vllm:prefix_cache_queries_total": 20.0,
                    "vllm:prefix_cache_hits_total": 15.0,
                    "vllm:prompt_tokens_cached_total": 100.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run", "artifact_files": ["server_reset.log"]}),
        encoding="utf-8",
    )
    (run_dir / "requests.jsonl").write_text(
        "\n".join(
            [
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
                                "ttft_ms": 10,
                                "e2e_ms": 20,
                                "cached_prompt_tokens": 8,
                                "prompt_tokens": 16,
                            },
                            "quality": {"score": 0.9},
                        },
                    }
                ),
                json.dumps(
                    {
                        "request": {
                            "request_id": "r2",
                            "prompt_layout": "document_before_instruction",
                            "task_type": "json",
                        },
                        "routing": {"adapter_id": "json"},
                        "response": {
                            "metrics": {
                                "ttft_ms": 12,
                                "e2e_ms": 24,
                                "cached_prompt_tokens": 4,
                                "prompt_tokens": 16,
                            },
                            "quality": {"score": 0.8},
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    table = build_adapter_cache_metrics(tmp_path)

    assert set(table["adapter_id"]) == {"qa", "json"}
    qa = table[table["adapter_id"].eq("qa")].iloc[0]
    assert qa["benchmark_cached_prompt_ratio"] == 0.5
    assert qa["server_prefix_cache_hit_rate"] == 0.75
    assert qa["server_cache_metric_scope"] == "per_condition_reset"


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
    assert summary.iloc[0]["quality_adjusted_goodput_ci95_half_width"] > 0
    assert summary.iloc[0]["quality_adjusted_goodput_ci95_low"] < 2.0
    assert summary.iloc[0]["quality_adjusted_goodput_ci95_high"] > 2.0


def test_benchmark_v0_summary_keeps_latest_complete_matrix():
    import pandas as pd

    rows = []
    for workload in [
        "shared_doc_qa",
        "mixed_tasks_same_doc",
        "prompt_layout_ablation",
        "low_overlap_control",
    ]:
        for router in ["semantic", "multitask", "sticky_session", "cache_aware", "oracle"]:
            for cache in ["standard_lora", "activated_lora", "copy_on_write"]:
                for seed in [17, 23, 31]:
                    rows.append(
                        {
                            "run_id": f"{workload}-{router}-{cache}-seed{seed}-100",
                            "request_count": 96,
                            "backend_kind": "mock",
                            "workload": workload,
                            "router_policy": router,
                            "cache_model": cache,
                            "quality_adjusted_goodput": 1.0,
                            "mean_quality": 0.8,
                            "p95_ttft_ms": 20.0,
                            "memory_token_footprint": 10,
                            "mean_ttft_ms": 10.0,
                            "p50_ttft_ms": 10.0,
                            "p99_ttft_ms": 30.0,
                            "mean_e2e_ms": 40.0,
                            "p95_e2e_ms": 50.0,
                            "slo_attainment_rate": 1.0,
                            "request_throughput": 2.0,
                            "token_throughput": 20.0,
                            "quality_adjusted_goodput_per_memory_token": 0.1,
                            "cache_hit_rate": 0.5,
                            "cached_prompt_token_ratio": 0.4,
                            "fragmentation_index": 1.0,
                            "eviction_count": 0,
                            "evicted_tokens": 0,
                        }
                    )
    rows.append({**rows[0], "run_id": "shared_doc_qa-semantic-standard_lora-seed17-200"})

    summary = benchmark_v0_summary(pd.DataFrame(rows))

    assert len(summary) == 180
    assert summary.iloc[0]["benchmark_suite"] == "benchmark_v0_mock"
    assert (
        summary[summary["run_id"].eq("shared_doc_qa-semantic-standard_lora-seed17-200")].shape[0]
        == 1
    )


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
