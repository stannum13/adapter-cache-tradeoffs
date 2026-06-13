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
    run_dir = run(config, run_id="unit")
    assert (run_dir / "requests.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config_resolved.yaml").exists()
