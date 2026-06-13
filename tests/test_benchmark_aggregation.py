import json

from specialization_cache_frontier.bench.aggregate import load_request_rows
from specialization_cache_frontier.bench.run_workload import run
from specialization_cache_frontier.config import (
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
    run_dir = run(config, run_id="unit", report_path=tmp_path / "report.md")
    assert (run_dir / "requests.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config_resolved.yaml").exists()


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
