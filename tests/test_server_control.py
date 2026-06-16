import json
import sys

import httpx
import pytest

from adapter_cache_bench.backends.server_control import (
    prepare_backend_server,
    wait_for_backend_server,
)
from adapter_cache_bench.bench.run_workload import run
from adapter_cache_bench.config import BackendConfig, BenchmarkConfig, WorkloadConfig


def test_prepare_backend_server_runs_reset_command(tmp_path):
    marker = tmp_path / "marker.txt"
    command = (
        f"{sys.executable} -c \"from pathlib import Path; Path({str(marker)!r}).write_text('ok')\""
    )
    config = BackendConfig(server_reset_command=command)

    artifact = prepare_backend_server(config, tmp_path)

    assert artifact == "server_reset.log"
    assert marker.read_text(encoding="utf-8") == "ok"
    assert "returncode: 0" in (tmp_path / "server_reset.log").read_text(encoding="utf-8")


def test_prepare_backend_server_raises_on_failed_reset(tmp_path):
    config = BackendConfig(
        server_reset_command=f'{sys.executable} -c "raise SystemExit(7)"',
    )

    with pytest.raises(RuntimeError):
        prepare_backend_server(config, tmp_path)

    assert "returncode: 7" in (tmp_path / "server_reset.log").read_text(encoding="utf-8")


def test_wait_for_backend_server_uses_warmup_url(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, timeout):
        calls["count"] += 1
        return httpx.Response(200)

    monkeypatch.setattr("adapter_cache_bench.backends.server_control.httpx.get", fake_get)

    wait_for_backend_server(
        BackendConfig(
            server_warmup_url="http://unit/health",
            server_warmup_timeout_s=1,
            server_warmup_interval_s=0.1,
        )
    )

    assert calls["count"] == 1


def test_benchmark_manifest_records_server_reset_artifact(tmp_path):
    marker = tmp_path / "reset.txt"
    command = (
        f"{sys.executable} -c \"from pathlib import Path; Path({str(marker)!r}).write_text('ok')\""
    )
    config = BenchmarkConfig(
        run_name="reset-unit",
        output_dir=str(tmp_path),
        workload=WorkloadConfig(name="shared_doc_qa", request_count=1, document_tokens=16),
        backend=BackendConfig(kind="mock", server_reset_command=command),
    )

    run_dir = run(config, run_id="reset-unit", generate_report_artifacts=False)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "server_reset.log" in manifest["artifact_files"]
    assert (run_dir / "server_reset.log").exists()
