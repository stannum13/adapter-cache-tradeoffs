import json

import pytest

from adapter_cache_bench.bench.sweep_state import (
    SweepChild,
    SweepOptions,
    artifact_complete,
    build_plan,
    child_id,
    execute_sweep,
    validate_budget,
)
from adapter_cache_bench.config import BenchmarkConfig, WorkloadConfig


def test_child_id_is_stable_for_sorted_dimensions():
    left = child_id("run", {"seed": 1, "strategy": "specialists"})
    right = child_id("run", {"strategy": "specialists", "seed": 1})

    assert left == right


def test_build_plan_records_children_and_request_count(tmp_path):
    config = BenchmarkConfig(run_name="parent", output_dir=str(tmp_path))
    child = BenchmarkConfig(
        run_name="child",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(request_count=7),
    )

    plan = build_plan(config, "matrix", [SweepChild(child, {"seed": 11})])

    assert plan["child_count"] == 1
    assert plan["planned_request_count"] == 7
    assert plan["children"][0]["run_name"] == "child"
    assert plan["children"][0]["dimensions"] == {"seed": 11}


def test_validate_budget_rejects_excessive_runs(tmp_path):
    children = [SweepChild(BenchmarkConfig(output_dir=str(tmp_path)), {}) for _ in range(2)]

    with pytest.raises(ValueError, match="planned run count"):
        validate_budget(children, SweepOptions(max_runs=1))


def test_artifact_complete_requires_complete_status(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in [
        "requests.jsonl",
        "summary.json",
        "config_resolved.yaml",
        "manifest.json",
    ]:
        (run_dir / name).write_text("{}", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "failed"}),
        encoding="utf-8",
    )

    assert artifact_complete(run_dir) is False
    (run_dir / "status.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    assert artifact_complete(run_dir) is True


def test_execute_sweep_dry_run_writes_plan_and_status(tmp_path):
    config = BenchmarkConfig(run_name="parent", output_dir=str(tmp_path))
    child = BenchmarkConfig(
        run_name="child",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(request_count=3),
    )

    status = execute_sweep(
        config=config,
        sweep_name="unit",
        children=[SweepChild(child, {"seed": 11})],
        run_child=lambda _config, _run_id: tmp_path / "unused",
        record_dimensions=lambda _path, _dimensions: None,
        options=SweepOptions(dry_run=True),
    )

    sweep_dir = tmp_path / "_sweeps" / "unit"
    assert (sweep_dir / "sweep_plan.json").exists()
    assert (sweep_dir / "sweep_status.json").exists()
    assert status["status"] == "dry_run"
    assert status["planned_request_count"] == 3


def test_execute_sweep_continue_on_error_records_failure(tmp_path):
    config = BenchmarkConfig(run_name="parent", output_dir=str(tmp_path))
    child = BenchmarkConfig(run_name="child", output_dir=str(tmp_path))

    status = execute_sweep(
        config=config,
        sweep_name="unit",
        children=[SweepChild(child, {"seed": 11})],
        run_child=lambda _config, _run_id: (_ for _ in ()).throw(RuntimeError("boom")),
        record_dimensions=lambda _path, _dimensions: None,
        options=SweepOptions(continue_on_error=True),
    )

    assert status["status"] == "complete_with_failures"
    assert status["counts"]["failed"] == 1
    assert status["children"][0]["exception_type"] == "RuntimeError"


def test_execute_sweep_resume_skips_complete_child(tmp_path):
    config = BenchmarkConfig(run_name="parent", output_dir=str(tmp_path))
    child = BenchmarkConfig(run_name="child", output_dir=str(tmp_path))
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    for name in [
        "requests.jsonl",
        "summary.json",
        "config_resolved.yaml",
        "manifest.json",
    ]:
        (run_dir / name).write_text("{}", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    calls = []

    status = execute_sweep(
        config=config,
        sweep_name="unit",
        children=[SweepChild(child, {"seed": 11})],
        run_child=lambda _config, _run_id: calls.append(_run_id) or run_dir,
        record_dimensions=lambda _path, _dimensions: None,
        options=SweepOptions(resume=True),
    )

    assert calls == []
    assert status["status"] == "complete"
    assert status["counts"]["skipped"] == 1
