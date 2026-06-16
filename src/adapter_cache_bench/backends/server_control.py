from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx

from adapter_cache_bench.config import BackendConfig


def reset_backend_server(config: BackendConfig, run_dir: Path) -> str | None:
    if not config.server_reset_command:
        return None
    result = subprocess.run(
        config.server_reset_command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        timeout=config.server_reset_timeout_s,
    )
    log_path = run_dir / "server_reset.log"
    log_path.write_text(
        "\n".join(
            [
                f"command: {config.server_reset_command}",
                f"returncode: {result.returncode}",
                "stdout:",
                result.stdout,
                "stderr:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Server reset command failed; see {log_path}")
    return log_path.name


def wait_for_backend_server(config: BackendConfig) -> None:
    if not config.server_warmup_url:
        return
    deadline = time.monotonic() + config.server_warmup_timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(config.server_warmup_url, timeout=5.0)
            if response.status_code < 500:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(max(0.1, config.server_warmup_interval_s))
    if last_error is not None:
        raise TimeoutError(
            f"Backend server did not become ready at {config.server_warmup_url}: {last_error}"
        )
    raise TimeoutError(f"Backend server did not become ready at {config.server_warmup_url}")


def prepare_backend_server(config: BackendConfig, run_dir: Path) -> str | None:
    reset_artifact = reset_backend_server(config, run_dir)
    wait_for_backend_server(config)
    return reset_artifact
