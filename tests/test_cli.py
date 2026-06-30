from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapter_cache_bench import cli
from adapter_cache_bench.config import BenchmarkConfig


def test_infer_runner_uses_workload_for_plain_serial_config() -> None:
    assert cli.infer_runner(BenchmarkConfig()) == "workload"


def test_infer_runner_uses_concurrent_for_plain_concurrent_config() -> None:
    config = BenchmarkConfig()
    config.backend.max_concurrency = 4

    assert cli.infer_runner(config) == "concurrent"


def test_infer_runner_uses_matrix_for_router_cache_matrix() -> None:
    config = BenchmarkConfig(matrix={"routers": ["random"], "caches": ["standard_lora"]})

    assert cli.infer_runner(config) == "matrix"


def test_infer_runner_uses_concurrency_sweep_for_strategy_concurrency_matrix() -> None:
    config = BenchmarkConfig(matrix={"strategies": ["base"], "concurrencies": [1, 2]})

    assert cli.infer_runner(config) == "concurrency-sweep"


def test_infer_runner_uses_exhaustive_sweep_for_model_matrix() -> None:
    config = BenchmarkConfig(matrix={"models": ["Qwen/Qwen2.5-1.5B-Instruct"]})

    assert cli.infer_runner(config) == "exhaustive-sweep"


def test_infer_runner_rejects_unknown_matrix_key() -> None:
    config = BenchmarkConfig(matrix={"surprise": [1]})

    with pytest.raises(ValueError, match="cannot infer runner"):
        cli.infer_runner(config)


def test_run_command_dispatches_workload_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: cli-smoke",
                f"output_dir: {tmp_path / 'runs'}",
                "workload:",
                "  request_count: 1",
            ]
        ),
        encoding="utf-8",
    )
    calls = {}

    def fake_run_workload(
        config: BenchmarkConfig,
        *,
        run_id: str | None = None,
        report_path: str | Path = "reports/adapter-cache-tradeoffs.md",
        tables_dir: str | Path = "reports/tables",
        generate_report_artifacts: bool = True,
    ) -> Path:
        calls["config"] = config
        calls["run_id"] = run_id
        calls["report_path"] = report_path
        calls["tables_dir"] = tables_dir
        calls["generate_report_artifacts"] = generate_report_artifacts
        return tmp_path / "runs" / "unit-run"

    monkeypatch.setattr(cli, "run_workload", fake_run_workload)

    result = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--run-id",
            "unit-run",
            "--report-path",
            "reports/unit.md",
            "--tables-dir",
            "reports/unit-tables",
            "--no-report",
        ]
    )

    assert result == 0
    assert calls["config"].run_name == "cli-smoke"
    assert calls["run_id"] == "unit-run"
    assert calls["report_path"] == "reports/unit.md"
    assert calls["tables_dir"] == "reports/unit-tables"
    assert calls["generate_report_artifacts"] is False
    assert str(tmp_path / "runs" / "unit-run") in capsys.readouterr().out


def test_run_command_matrix_dry_run_writes_sweep_state(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: cli-matrix",
                f"output_dir: {runs_dir}",
                "workload:",
                "  request_count: 2",
                "matrix:",
                "  routers: [random, cache_aware]",
                "  caches: [standard_lora]",
                "  workloads: [shared_doc_qa]",
                "  seeds: [1]",
            ]
        ),
        encoding="utf-8",
    )

    result = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--sweep-name",
            "unit-sweep",
            "--dry-run",
            "--max-runs",
            "2",
            "--max-requests",
            "4",
        ]
    )

    assert result == 0
    status_path = runs_dir / "_sweeps" / "unit-sweep" / "sweep_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "dry_run"
    assert status["child_count"] == 2
    assert status["planned_request_count"] == 4
    assert status["budget"]["planned_runs"] == 2


def test_report_command_calls_generate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = {}

    def fake_generate_report(
        runs_dir: str | Path,
        report_path: str | Path = "reports/adapter-cache-tradeoffs.md",
        tables_dir: str | Path = "reports/tables",
        figures_dir: str | Path = "reports/figures",
    ) -> Path:
        calls["runs_dir"] = runs_dir
        calls["report_path"] = report_path
        calls["tables_dir"] = tables_dir
        calls["figures_dir"] = figures_dir
        return tmp_path / "report.md"

    monkeypatch.setattr(cli, "generate_report", fake_generate_report)

    result = cli.main(
        [
            "report",
            "--runs-dir",
            "runs",
            "--report-path",
            "out.md",
            "--tables-dir",
            "tables",
            "--figures-dir",
            "figures",
        ]
    )

    assert result == 0
    assert calls == {
        "runs_dir": "runs",
        "report_path": "out.md",
        "tables_dir": "tables",
        "figures_dir": "figures",
    }
    assert str(tmp_path / "report.md") in capsys.readouterr().out


def test_bundle_command_calls_build_evidence_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = {}

    def fake_build_evidence_bundle(
        *,
        bundle_name: str,
        runs_dir: str | Path = "artifacts/runs",
        output_dir: str | Path | None = None,
        run_ids: list[str] | None = None,
        run_globs: list[str] | None = None,
        reports: list[str | Path] | None = None,
        figures: list[str | Path] | None = None,
        tables: list[str | Path] | None = None,
        repo_dir: str | Path = ".",
    ) -> Path:
        calls["bundle_name"] = bundle_name
        calls["runs_dir"] = runs_dir
        calls["output_dir"] = output_dir
        calls["run_ids"] = run_ids
        calls["run_globs"] = run_globs
        calls["reports"] = reports
        calls["figures"] = figures
        calls["tables"] = tables
        calls["repo_dir"] = repo_dir
        return tmp_path / "bundle_manifest.json"

    monkeypatch.setattr(cli, "build_evidence_bundle", fake_build_evidence_bundle)

    result = cli.main(
        [
            "bundle",
            "--bundle-name",
            "release",
            "--runs-dir",
            "runs",
            "--output-dir",
            "evidence/release",
            "--run",
            "run-a",
            "--run-glob",
            "run-*",
            "--report",
            "reports/out.md",
            "--figure",
            "reports/figures/a.png",
            "--table",
            "reports/tables/claim_evidence.csv",
            "--repo-dir",
            ".",
        ]
    )

    assert result == 0
    assert calls == {
        "bundle_name": "release",
        "runs_dir": "runs",
        "output_dir": "evidence/release",
        "run_ids": ["run-a"],
        "run_globs": ["run-*"],
        "reports": ["reports/out.md"],
        "figures": ["reports/figures/a.png"],
        "tables": ["reports/tables/claim_evidence.csv"],
        "repo_dir": ".",
    }
    assert str(tmp_path / "bundle_manifest.json") in capsys.readouterr().out


def test_doctor_command_reports_vllm_bridge_budget(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "--max-runs",
            "12",
            "--max-requests",
            "300",
            "--estimated-seconds-per-run",
            "180",
            "--max-estimated-gpu-hours",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "status: ok" in output
    assert "runner: exhaustive-sweep" in output
    assert "planned runs: 12" in output
    assert "planned requests: 288" in output
    assert "estimated GPU hours: 0.600" in output
    assert "non-warm cache conditions on remote backends" in output


def test_doctor_command_returns_error_for_budget_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: cli-matrix",
                f"output_dir: {tmp_path / 'runs'}",
                "workload:",
                "  request_count: 2",
                "matrix:",
                "  routers: [random, cache_aware]",
                "  caches: [standard_lora]",
                "  workloads: [shared_doc_qa]",
                "  seeds: [1]",
            ]
        ),
        encoding="utf-8",
    )

    result = cli.main(["doctor", "--config", str(config_path), "--max-runs", "1"])

    output = capsys.readouterr().out
    assert result == 1
    assert "status: error" in output
    assert "planned run count 2 exceeds --max-runs 1" in output


def test_doctor_command_can_check_gcloud_without_starting_resources(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")

    def fake_run_text_command(command: list[str]) -> tuple[int, str]:
        joined = " ".join(command)
        if "auth list" in joined:
            return 0, "user@example.com"
        if "get-value project" in joined:
            return 0, "project-a"
        if "get-value compute/zone" in joined:
            return 0, "us-central1-a"
        raise AssertionError(command)

    monkeypatch.setattr(cli, "_run_text_command", fake_run_text_command)

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "--check-gcloud",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "gcloud account detected: user@example.com" in output
    assert "gcloud project detected: project-a" in output
    assert "gcloud zone detected: us-central1-a" in output
