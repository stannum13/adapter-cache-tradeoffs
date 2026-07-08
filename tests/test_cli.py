from __future__ import annotations

import json
import subprocess
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
        strict: bool = False,
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
        calls["strict"] = strict
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
            "--strict",
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
        "strict": True,
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


def test_doctor_command_reports_success_json(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "--json",
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
    payload = json.loads(output)
    assert result == 0
    assert output.count("\n") == 1
    assert payload["status"] == "ok"
    assert payload["runner"] == "exhaustive-sweep"
    assert payload["planned_runs"] == 12
    assert payload["planned_requests"] == 288
    assert payload["estimated_gpu_hours"] == pytest.approx(0.6)
    assert payload["errors"] == []
    assert any(
        "non-warm cache conditions on remote backends" in warning for warning in payload["warnings"]
    )


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


def test_doctor_command_reports_error_json(
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

    result = cli.main(["doctor", "--config", str(config_path), "--json", "--max-runs", "1"])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert result == 1
    assert output.count("\n") == 1
    assert payload == {
        "status": "error",
        "runner": "matrix",
        "planned_runs": 2,
        "planned_requests": 4,
        "estimated_gpu_hours": None,
        "warnings": [],
        "errors": ["planned run count 2 exceeds --max-runs 1"],
    }


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


def test_run_text_command_ignores_successful_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["gcloud"],
            returncode=0,
            stdout='{"status": "TERMINATED"}\n',
            stderr="WARNING: optional component update available\n",
        )

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    code, output = cli._run_text_command(["gcloud", "compute", "instances", "describe"])

    assert code == 0
    assert output == '{"status": "TERMINATED"}'


def test_doctor_command_does_not_request_gcloud_check_when_already_checked(
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
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--check-gcloud",
            "--gcloud-zone",
            "us-central1-b",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "gcloud preflight target zone: us-central1-b" in output
    assert "rerun with --check-gcloud" not in output


def test_doctor_command_can_describe_gcloud_instance_without_starting_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")

    commands: list[list[str]] = []

    def fake_run_text_command(command: list[str]) -> tuple[int, str]:
        commands.append(command)
        if "describe" in command:
            return (
                0,
                json.dumps(
                    {
                        "status": "TERMINATED",
                        "machineType": (
                            "https://www.googleapis.com/compute/v1/projects/project-a/"
                            "zones/us-central1-b/machineTypes/g2-standard-8"
                        ),
                        "labels": {"ttl_hours": "8", "purpose": "benchmark"},
                    }
                ),
            )
        raise AssertionError(command)

    monkeypatch.setattr(cli, "_run_text_command", fake_run_text_command)

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--gcloud-instance",
            "adapter-cache-vllm-l4-b",
            "--gcloud-project",
            "project-a",
            "--gcloud-zone",
            "us-central1-b",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "gcloud instance adapter-cache-vllm-l4-b status: TERMINATED" in output
    assert "starting it will incur GPU costs" in output
    assert "gcloud instance adapter-cache-vllm-l4-b machine type: g2-standard-8" in output
    assert "gcloud instance adapter-cache-vllm-l4-b ttl_hours label: 8" in output
    assert commands == [
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            "adapter-cache-vllm-l4-b",
            "--format=json",
            "--project=project-a",
            "--zone=us-central1-b",
        ]
    ]


def test_doctor_command_warns_when_gcloud_instance_is_running_without_ttl(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(
        cli,
        "_run_text_command",
        lambda command: (
            0,
            json.dumps(
                {
                    "status": "RUNNING",
                    "machineType": "zones/us-central1-b/machineTypes/g2-standard-8",
                    "labels": {},
                }
            ),
        ),
    )

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--gcloud-instance",
            "adapter-cache-vllm-l4-b",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "gcloud instance adapter-cache-vllm-l4-b status: RUNNING" in output
    assert "already running; check for stale servers" in output
    assert "has no ttl_hours label" in output


def test_doctor_command_fails_for_inaccessible_gcloud_instance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(cli, "_run_text_command", lambda command: (1, "not found"))

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--gcloud-instance",
            "missing-vm",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "gcloud instance is not accessible: missing-vm" in output


def test_doctor_command_requires_cloud_provenance_for_cloud_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in cli.REQUIRED_CLOUD_PROVENANCE_VARS:
        monkeypatch.delenv(name, raising=False)

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--require-cloud-provenance",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "missing cloud provenance environment variable(s)" in output
    assert "ACB_CLOUD_PROJECT" in output


def test_doctor_command_cross_checks_cloud_provenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = {
        "ACB_CLOUD_PROVIDER": "gcp",
        "ACB_CLOUD_PROJECT": "project-a",
        "ACB_CLOUD_ZONE": "us-central1-b",
        "ACB_CLOUD_INSTANCE": "adapter-cache-vllm-l4-b",
        "ACB_CLOUD_MACHINE_TYPE": "g2-standard-8",
        "ACB_CLOUD_GPU_TYPE": "nvidia-l4",
        "ACB_CLOUD_GPU_COUNT": "1",
        "ACB_CLOUD_TTL_HOURS": "8",
        "ACB_VLLM_IMAGE": "vllm/vllm-openai:latest",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--gcloud-project",
            "wrong-project",
            "--gcloud-zone",
            "us-central1-b",
            "--require-cloud-provenance",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "cloud provenance environment is present" in output
    assert "ACB_CLOUD_PROJECT=project-a does not match preflight value wrong-project" in output


def test_doctor_command_cross_checks_cloud_provenance_against_instance_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = {
        "ACB_CLOUD_PROVIDER": "gcp",
        "ACB_CLOUD_PROJECT": "project-a",
        "ACB_CLOUD_ZONE": "us-central1-b",
        "ACB_CLOUD_INSTANCE": "adapter-cache-vllm-l4-b",
        "ACB_CLOUD_MACHINE_TYPE": "wrong-machine",
        "ACB_CLOUD_GPU_TYPE": "nvidia-l4",
        "ACB_CLOUD_GPU_COUNT": "2",
        "ACB_CLOUD_TTL_HOURS": "8",
        "ACB_VLLM_IMAGE": "vllm/vllm-openai:latest",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(
        cli,
        "_run_text_command",
        lambda command: (
            0,
            json.dumps(
                {
                    "status": "TERMINATED",
                    "machineType": "zones/us-central1-b/machineTypes/g2-standard-8",
                    "guestAccelerators": [
                        {
                            "acceleratorType": "zones/us-central1-b/acceleratorTypes/nvidia-l4",
                            "acceleratorCount": 1,
                        }
                    ],
                    "labels": {"ttl_hours": "8"},
                }
            ),
        ),
    )

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--gcloud-instance",
            "adapter-cache-vllm-l4-b",
            "--gcloud-project",
            "project-a",
            "--gcloud-zone",
            "us-central1-b",
            "--require-cloud-provenance",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert (
        "ACB_CLOUD_MACHINE_TYPE=wrong-machine does not match preflight value g2-standard-8"
        in output
    )
    assert "ACB_CLOUD_GPU_COUNT=2 does not match preflight value 1" in output


def test_doctor_command_requires_instance_metadata_for_gcloud_quota_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--gcloud-zone",
            "us-central1-b",
            "--check-gcloud-quota",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "gcloud quota check requires --gcloud-instance with GPU metadata" in output


def test_doctor_command_blocks_when_project_gpu_quota_has_no_headroom(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")

    def fake_run_text_command(command: list[str]) -> tuple[int, str]:
        joined = " ".join(command)
        if "instances describe" in joined:
            return (
                0,
                json.dumps(
                    {
                        "status": "TERMINATED",
                        "machineType": "zones/us-central1-b/machineTypes/g2-standard-8",
                        "guestAccelerators": [
                            {
                                "acceleratorType": (
                                    "zones/us-central1-b/acceleratorTypes/nvidia-l4"
                                ),
                                "acceleratorCount": 1,
                            }
                        ],
                        "labels": {"ttl_hours": "8"},
                    }
                ),
            )
        if "regions describe us-central1" in joined:
            return 0, json.dumps({"quotas": [{"metric": "NVIDIA_L4_GPUS", "limit": 1, "usage": 0}]})
        if "project-info describe" in joined:
            return 0, json.dumps(
                {"quotas": [{"metric": "GPUS_ALL_REGIONS", "limit": 1, "usage": 1}]}
            )
        raise AssertionError(command)

    monkeypatch.setattr(cli, "_run_text_command", fake_run_text_command)

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
            "--gcloud-instance",
            "adapter-cache-vllm-l4-b",
            "--gcloud-project",
            "project-a",
            "--gcloud-zone",
            "us-central1-b",
            "--check-gcloud-quota",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "regional GPU quota NVIDIA_L4_GPUS headroom: 1" in output
    assert "project GPU quota GPUS_ALL_REGIONS headroom 0 is below required 1" in output


def _fake_redaction_gcloud_command(command: list[str]) -> tuple[int, str]:
    joined = " ".join(command)
    if "auth list" in joined:
        return 0, "sensitive.user@example.com"
    if "config get-value project" in joined:
        return 0, "secret-project"
    if "config get-value compute/zone" in joined:
        return 0, "secret-zone-a"
    if "instances describe" in joined:
        return (
            0,
            json.dumps(
                {
                    "status": "TERMINATED",
                    "machineType": "zones/secret-zone-a/machineTypes/g2-standard-8",
                    "guestAccelerators": [
                        {
                            "acceleratorType": ("zones/secret-zone-a/acceleratorTypes/nvidia-l4"),
                            "acceleratorCount": 1,
                        }
                    ],
                    "labels": {"ttl_hours": "8"},
                }
            ),
        )
    if "regions describe" in joined:
        return 0, json.dumps({"quotas": [{"metric": "NVIDIA_L4_GPUS", "limit": 1, "usage": 0}]})
    if "project-info describe" in joined:
        return 0, json.dumps({"quotas": [{"metric": "GPUS_ALL_REGIONS", "limit": 1, "usage": 1}]})
    raise AssertionError(command)


def _redacted_doctor_args(*extra_args: str) -> list[str]:
    return [
        "doctor",
        "--config",
        "configs/benchmark/vllm_bridge_reset.yaml",
        "configs/benchmark/gcloud_7b_lora_bridge_reset.yaml",
        "--check-gcloud",
        "--gcloud-instance",
        "secret-instance",
        "--gcloud-project",
        "secret-project",
        "--gcloud-zone",
        "secret-zone-a",
        "--check-gcloud-quota",
        *extra_args,
        "--redact",
    ]


def test_doctor_command_redacts_gcloud_text_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(cli, "_run_text_command", _fake_redaction_gcloud_command)

    result = cli.main(_redacted_doctor_args())

    output = capsys.readouterr().out
    assert result == 1
    assert "sensitive.user@example.com" not in output
    assert "secret-project" not in output
    assert "secret-zone-a" not in output
    assert "secret-instance" not in output
    assert "g2-standard-8" not in output
    assert "nvidia-l4 x1" not in output
    assert "gcloud account detected: <redacted-account>" in output
    assert "gcloud project detected: <redacted-project>" in output
    assert "gcloud zone detected: <redacted-zone>" in output
    assert "gcloud instance <redacted-instance> is stopped" in output
    assert "project GPU quota GPUS_ALL_REGIONS headroom 0 is below required 1" in output


def test_doctor_command_redacts_gcloud_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(cli, "_run_text_command", _fake_redaction_gcloud_command)

    result = cli.main(_redacted_doctor_args("--json"))

    payload = json.loads(capsys.readouterr().out)
    payload_text = json.dumps(payload)
    assert result == 1
    assert payload["status"] == "error"
    assert payload["planned_runs"] == 12
    assert "sensitive.user@example.com" not in payload_text
    assert "secret-project" not in payload_text
    assert "secret-zone-a" not in payload_text
    assert "secret-instance" not in payload_text
    assert "g2-standard-8" not in payload_text
    assert "nvidia-l4 x1" not in payload_text
    assert "gcloud account detected: <redacted-account>" in payload["warnings"]
    assert "gcloud preflight target zone: <redacted-zone>" in payload["warnings"]
    assert "project GPU quota GPUS_ALL_REGIONS headroom 0 is below required 1" in payload["errors"]


def test_doctor_redaction_preserves_gcloud_command_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/gcloud")

    def fake_run_text_command(command: list[str]) -> tuple[int, str]:
        commands.append(command)
        return _fake_redaction_gcloud_command(command)

    monkeypatch.setattr(cli, "_run_text_command", fake_run_text_command)

    result = cli.main(_redacted_doctor_args())

    output = capsys.readouterr().out
    assert result == 1
    assert "secret-project" not in output
    assert "secret-zone-a" not in output
    assert "secret-instance" not in output
    assert [
        "gcloud",
        "compute",
        "instances",
        "describe",
        "secret-instance",
        "--format=json",
        "--project=secret-project",
        "--zone=secret-zone-a",
    ] in commands
    assert [
        "gcloud",
        "compute",
        "regions",
        "describe",
        "secret-zone",
        "--format=json",
        "--project=secret-project",
    ] in commands
    assert [
        "gcloud",
        "compute",
        "project-info",
        "describe",
        "--format=json",
        "--project=secret-project",
    ] in commands


def test_doctor_redacts_cloud_provenance_mismatch_values() -> None:
    message = "ACB_CLOUD_PROJECT=secret-project does not match preflight value other-secret-project"

    assert cli._redact_doctor_message(message) == (
        "ACB_CLOUD_PROJECT=<redacted-value> does not match preflight value <redacted-value>"
    )


def test_doctor_redacts_unsupported_accelerator_value() -> None:
    message = "gcloud quota check does not support accelerator internal-accelerator-type"

    assert cli._redact_doctor_message(message) == (
        "gcloud quota check does not support accelerator <redacted-accelerator>"
    )


def test_doctor_redacts_region_without_hiding_json_parse_cause() -> None:
    message = "gcloud region quota is not accessible: secret-region: invalid JSON"

    assert cli._redact_doctor_message(message) == (
        "gcloud region quota is not accessible: <redacted-region>: invalid JSON"
    )


def test_doctor_command_blocks_when_local_tunnel_port_is_bound(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_local_port_available", lambda port: False)

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "--check-local-port",
            "--local-port",
            "8000",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "local tunnel port 8000 is already in use" in output


def test_doctor_command_rejects_invalid_local_port(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "doctor",
                "--config",
                "configs/benchmark/vllm_bridge_reset.yaml",
                "--check-local-port",
                "--local-port",
                "70000",
            ]
        )

    assert exc_info.value.code == 2
    assert "port must be between 1 and 65535" in capsys.readouterr().err


def test_doctor_command_reports_malformed_backend_url_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "bad_url.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: cli-bad-url",
                "backend:",
                "  kind: vllm",
                "  base_url: http://localhost:bad/v1",
                "  stream: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_local_port_available", lambda port: True)

    result = cli.main(
        [
            "doctor",
            "--config",
            str(config_path),
            "--check-local-port",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "cli-bad-url: invalid URL port in http://localhost:bad/v1" in output


def test_doctor_command_redacts_malformed_backend_url_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "bad_url.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: cli-bad-url",
                "backend:",
                "  kind: vllm",
                "  base_url: http://private-host.internal:bad/v1",
                "  stream: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_local_port_available", lambda port: True)

    result = cli.main(
        [
            "doctor",
            "--config",
            str(config_path),
            "--check-local-port",
            "--redact",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "private-host.internal" not in output
    assert "cli-bad-url: invalid URL port in <redacted-url>" in output


def test_doctor_command_accepts_local_port_overlay(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_local_port_available", lambda port: True)

    result = cli.main(
        [
            "doctor",
            "--config",
            "configs/benchmark/vllm_bridge_reset.yaml",
            "configs/benchmark/local_port_8001.yaml",
            "--check-local-port",
            "--local-port",
            "8001",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "local tunnel port 8001 is available" in output
    assert "backend URLs use local tunnel port 8001" in output
