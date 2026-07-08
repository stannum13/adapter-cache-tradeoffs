from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from adapter_cache_bench.analysis.evidence_bundle import (
    EvidenceBundleValidationError,
    build_evidence_bundle,
)
from adapter_cache_bench.analysis.report import generate_report
from adapter_cache_bench.bench.run_concurrency_sweep import expand_concurrency_sweep_children
from adapter_cache_bench.bench.run_concurrent import run_concurrent
from adapter_cache_bench.bench.run_exhaustive_sweep import expand_exhaustive_sweep
from adapter_cache_bench.bench.run_matrix import expand_matrix_sweep
from adapter_cache_bench.bench.run_workload import run as run_workload
from adapter_cache_bench.bench.sweep_state import (
    SweepChild,
    SweepOptions,
    add_sweep_arguments,
    execute_sweep,
    options_from_args,
    record_sweep_dimensions,
    validate_budget,
)
from adapter_cache_bench.config import BenchmarkConfig, load_config

RunnerName = Literal[
    "workload",
    "concurrent",
    "matrix",
    "concurrency-sweep",
    "exhaustive-sweep",
]

RUNNER_CHOICES: tuple[str, ...] = (
    "auto",
    "workload",
    "concurrent",
    "matrix",
    "concurrency-sweep",
    "exhaustive-sweep",
)
SWEEP_RUNNERS = {"matrix", "concurrency-sweep", "exhaustive-sweep"}
MATRIX_KEYS = {"routers", "caches", "cache_conditions", "workloads", "seeds"}
CONCURRENCY_SWEEP_KEYS = {"strategies", "concurrencies", "cache_conditions", "seeds"}
EXHAUSTIVE_SWEEP_KEYS = {
    "strategies",
    "concurrencies",
    "workloads",
    "caches",
    "cache_conditions",
    "seeds",
    "overlap_fractions",
    "adapter_counts",
    "tenants",
    "isolation_scopes",
    "models",
}
KNOWN_MATRIX_KEYS = MATRIX_KEYS | CONCURRENCY_SWEEP_KEYS | EXHAUSTIVE_SWEEP_KEYS
EXHAUSTIVE_ONLY_KEYS = EXHAUSTIVE_SWEEP_KEYS - CONCURRENCY_SWEEP_KEYS - MATRIX_KEYS
SWEEP_OPTION_NAMES = (
    "sweep_name",
    "resume",
    "continue_on_error",
    "dry_run",
    "max_runs",
    "max_requests",
    "estimated_seconds_per_run",
    "max_estimated_gpu_hours",
)
REMOTE_BACKENDS = {"vllm", "openai_compatible"}
REQUIRED_CLOUD_PROVENANCE_VARS = (
    "ACB_CLOUD_PROVIDER",
    "ACB_CLOUD_PROJECT",
    "ACB_CLOUD_ZONE",
    "ACB_CLOUD_INSTANCE",
    "ACB_CLOUD_MACHINE_TYPE",
    "ACB_CLOUD_GPU_TYPE",
    "ACB_CLOUD_GPU_COUNT",
    "ACB_CLOUD_TTL_HOURS",
    "ACB_VLLM_IMAGE",
)
GPU_QUOTA_METRIC_BY_ACCELERATOR = {
    "nvidia-l4": "NVIDIA_L4_GPUS",
}


@dataclass(frozen=True)
class DoctorResult:
    runner: RunnerName
    planned_runs: int
    planned_requests: int
    estimated_gpu_hours: float | None
    warnings: list[str]
    errors: list[str]


def infer_runner(config: BenchmarkConfig) -> RunnerName:
    matrix_keys = set(config.matrix)
    if not matrix_keys:
        if config.backend.max_concurrency > 1 or config.backend.request_spacing_ms > 0:
            return "concurrent"
        return "workload"

    unknown_keys = matrix_keys - KNOWN_MATRIX_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"cannot infer runner for unknown matrix key(s): {unknown}; pass --runner")

    if matrix_keys & EXHAUSTIVE_ONLY_KEYS:
        return "exhaustive-sweep"
    if "strategies" in matrix_keys or "concurrencies" in matrix_keys:
        if matrix_keys <= CONCURRENCY_SWEEP_KEYS:
            return "concurrency-sweep"
        return "exhaustive-sweep"
    if matrix_keys <= MATRIX_KEYS:
        return "matrix"

    keys = ", ".join(sorted(matrix_keys))
    raise ValueError(f"cannot infer runner for matrix key combination: {keys}; pass --runner")


def _runner_children(config: BenchmarkConfig, runner: RunnerName) -> list[SweepChild]:
    if runner in {"workload", "concurrent"}:
        return [SweepChild(config, {})]
    if runner == "matrix":
        return expand_matrix_sweep(config)
    if runner == "concurrency-sweep":
        return expand_concurrency_sweep_children(config)
    if runner == "exhaustive-sweep":
        return [
            SweepChild(child_config, dimensions)
            for child_config, dimensions in expand_exhaustive_sweep(config)
        ]
    raise AssertionError(f"unhandled runner: {runner}")


def _doctor_sweep_options(args: argparse.Namespace) -> SweepOptions:
    return SweepOptions(
        max_runs=args.max_runs,
        max_requests=args.max_requests,
        estimated_seconds_per_run=args.estimated_seconds_per_run,
        max_estimated_gpu_hours=args.max_estimated_gpu_hours,
    )


def _run_command(command: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return 127, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _run_text_command(command: list[str]) -> tuple[int, str]:
    code, stdout, stderr = _run_command(command)
    if code == 0:
        return code, stdout
    output = "\n".join(part for part in [stdout, stderr] if part)
    return code, output


def _last_path_part(value: object) -> str:
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def _region_from_zone(zone: str | None) -> str | None:
    if not zone or zone.count("-") < 2:
        return None
    return zone.rsplit("-", 1)[0]


def _quota_headroom(payload: dict[str, object], metric_name: str) -> float | None:
    quotas = payload.get("quotas", [])
    if not isinstance(quotas, list):
        return None
    for quota in quotas:
        if not isinstance(quota, dict) or quota.get("metric") != metric_name:
            continue
        try:
            return float(quota.get("limit", 0.0)) - float(quota.get("usage", 0.0))
        except (TypeError, ValueError):
            return None
    return None


def _json_gcloud_command(
    command: list[str],
    *,
    errors: list[str],
    error_label: str,
) -> dict[str, object] | None:
    code, payload_text = _run_text_command(command)
    if code != 0:
        errors.append(error_label)
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        errors.append(f"{error_label}: invalid JSON")
        return None
    return payload if isinstance(payload, dict) else {}


def _check_gcloud_config(warnings: list[str], errors: list[str]) -> None:
    if shutil.which("gcloud") is None:
        errors.append("gcloud is not installed or not on PATH")
        return

    account_code, accounts = _run_text_command(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"]
    )
    if account_code != 0 or not accounts.strip():
        errors.append("gcloud has no authenticated account")
    else:
        first_account = accounts.splitlines()[0]
        warnings.append(f"gcloud account detected: {first_account}")

    project_code, project = _run_text_command(["gcloud", "config", "get-value", "project"])
    if project_code != 0 or not project.strip():
        errors.append("gcloud project is not configured")
    else:
        warnings.append(f"gcloud project detected: {project.splitlines()[0]}")

    zone_code, zone = _run_text_command(["gcloud", "config", "get-value", "compute/zone"])
    if zone_code != 0 or not zone.strip():
        warnings.append("gcloud compute/zone is not configured")
    else:
        warnings.append(f"gcloud zone detected: {zone.splitlines()[0]}")


def _check_gcloud_instance(
    warnings: list[str],
    errors: list[str],
    *,
    instance: str,
    project: str | None,
    zone: str | None,
) -> dict[str, object] | None:
    if shutil.which("gcloud") is None:
        errors.append("gcloud is not installed or not on PATH")
        return None
    command = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        instance,
        "--format=json",
    ]
    if project:
        command.append(f"--project={project}")
    if zone:
        command.append(f"--zone={zone}")
    describe_code, payload_text = _run_text_command(command)
    if describe_code != 0:
        errors.append(f"gcloud instance is not accessible: {instance}")
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        errors.append(f"gcloud instance describe returned invalid JSON: {instance}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"gcloud instance describe returned non-object JSON: {instance}")
        return None
    status = str(payload.get("status", "unknown"))
    machine_type = _last_path_part(payload.get("machineType", "unknown"))
    labels = payload.get("labels", {})
    if not isinstance(labels, dict):
        labels = {}
    accelerators = payload.get("guestAccelerators", [])
    if isinstance(accelerators, list):
        for accelerator in accelerators:
            if not isinstance(accelerator, dict):
                continue
            accelerator_type = _last_path_part(accelerator.get("acceleratorType", "unknown"))
            accelerator_count = accelerator.get("acceleratorCount", "unknown")
            warnings.append(
                f"gcloud instance {instance} accelerator: {accelerator_type} x{accelerator_count}"
            )
    warnings.append(f"gcloud instance {instance} status: {status}")
    warnings.append(f"gcloud instance {instance} machine type: {machine_type}")
    if status == "RUNNING":
        warnings.append(f"gcloud instance {instance} is already running; check for stale servers")
    elif status == "TERMINATED":
        warnings.append(f"gcloud instance {instance} is stopped; starting it will incur GPU costs")
    else:
        warnings.append(f"gcloud instance {instance} has non-standard status: {status}")
    ttl_hours = labels.get("ttl_hours")
    if ttl_hours:
        warnings.append(f"gcloud instance {instance} ttl_hours label: {ttl_hours}")
    else:
        warnings.append(f"gcloud instance {instance} has no ttl_hours label")
    return payload


def _required_gpu_count(instance_payload: dict[str, object] | None) -> tuple[str | None, int]:
    if not instance_payload:
        return None, 0
    accelerators = instance_payload.get("guestAccelerators", [])
    if not isinstance(accelerators, list):
        return None, 0
    total = 0
    accelerator_type = None
    for accelerator in accelerators:
        if not isinstance(accelerator, dict):
            continue
        accelerator_type = _last_path_part(accelerator.get("acceleratorType", "unknown"))
        try:
            total += int(accelerator.get("acceleratorCount", 0))
        except (TypeError, ValueError):
            continue
    return accelerator_type, total


def _check_gcloud_quota(
    warnings: list[str],
    errors: list[str],
    *,
    project: str | None,
    zone: str | None,
    instance_payload: dict[str, object] | None,
) -> None:
    if shutil.which("gcloud") is None:
        errors.append("gcloud is not installed or not on PATH")
        return
    if not instance_payload:
        errors.append("gcloud quota check requires --gcloud-instance with GPU metadata")
        return
    accelerator_type, required_gpu_count = _required_gpu_count(instance_payload)
    if not accelerator_type or required_gpu_count <= 0:
        errors.append("gcloud quota check requires an instance with GPU accelerator metadata")
        return
    regional_metric = GPU_QUOTA_METRIC_BY_ACCELERATOR.get(accelerator_type)
    if not regional_metric:
        errors.append(f"gcloud quota check does not support accelerator {accelerator_type}")
        return

    region = _region_from_zone(zone)
    if not region:
        errors.append("gcloud quota check requires --gcloud-zone")
        return

    region_command = ["gcloud", "compute", "regions", "describe", region, "--format=json"]
    project_command = ["gcloud", "compute", "project-info", "describe", "--format=json"]
    if project:
        region_command.append(f"--project={project}")
        project_command.append(f"--project={project}")
    region_payload = _json_gcloud_command(
        region_command,
        errors=errors,
        error_label=f"gcloud region quota is not accessible: {region}",
    )
    project_payload = _json_gcloud_command(
        project_command,
        errors=errors,
        error_label="gcloud project quota is not accessible",
    )
    if region_payload:
        headroom = _quota_headroom(region_payload, regional_metric)
        if headroom is None:
            warnings.append(f"regional quota {regional_metric} is unavailable")
        elif headroom < required_gpu_count:
            errors.append(
                f"regional GPU quota {regional_metric} headroom {headroom:g} "
                f"is below required {required_gpu_count}"
            )
        else:
            warnings.append(f"regional GPU quota {regional_metric} headroom: {headroom:g}")
    if project_payload:
        headroom = _quota_headroom(project_payload, "GPUS_ALL_REGIONS")
        if headroom is None:
            warnings.append("project quota GPUS_ALL_REGIONS is unavailable")
        elif headroom < required_gpu_count:
            errors.append(
                f"project GPU quota GPUS_ALL_REGIONS headroom {headroom:g} "
                f"is below required {required_gpu_count}"
            )
        else:
            warnings.append(f"project GPU quota GPUS_ALL_REGIONS headroom: {headroom:g}")


def _local_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _check_local_port(warnings: list[str], errors: list[str], *, port: int) -> None:
    if _local_port_available(port):
        warnings.append(f"local tunnel port {port} is available")
    else:
        errors.append(f"local tunnel port {port} is already in use")


def _url_port(value: str) -> int | None:
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid URL port in {value}") from exc
    if port is not None:
        return port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    return None


def _check_backend_url_ports(
    warnings: list[str],
    errors: list[str],
    *,
    children: list[SweepChild],
    local_port: int,
) -> None:
    ports: set[int] = set()
    for child in children:
        backend = child.config.backend
        if backend.kind not in REMOTE_BACKENDS:
            continue
        for value in [backend.base_url, backend.metrics_url, backend.server_warmup_url or ""]:
            if value:
                try:
                    port = _url_port(value)
                except ValueError as exc:
                    errors.append(f"{child.config.run_name}: {exc}")
                    continue
                if port is not None:
                    ports.add(port)
    if not ports:
        return
    if ports == {local_port}:
        warnings.append(f"backend URLs use local tunnel port {local_port}")
    else:
        warnings.append(
            f"backend URL ports {', '.join(str(item) for item in sorted(ports))} "
            f"do not all match --local-port {local_port}"
        )


def _check_cloud_provenance(
    warnings: list[str],
    errors: list[str],
    *,
    gcloud_instance: str | None,
    gcloud_project: str | None,
    gcloud_zone: str | None,
    instance_payload: dict[str, object] | None,
) -> None:
    missing = [name for name in REQUIRED_CLOUD_PROVENANCE_VARS if not os.environ.get(name)]
    if missing:
        errors.append("missing cloud provenance environment variable(s): " + ", ".join(missing))
        return

    provider = os.environ["ACB_CLOUD_PROVIDER"]
    if provider != "gcp":
        errors.append(f"ACB_CLOUD_PROVIDER must be gcp for this path, got {provider}")
    expected = {
        "ACB_CLOUD_PROJECT": gcloud_project,
        "ACB_CLOUD_ZONE": gcloud_zone,
        "ACB_CLOUD_INSTANCE": gcloud_instance,
    }
    if instance_payload:
        labels = instance_payload.get("labels", {})
        if not isinstance(labels, dict):
            labels = {}
        accelerator_type, gpu_count = _required_gpu_count(instance_payload)
        expected["ACB_CLOUD_MACHINE_TYPE"] = _last_path_part(
            instance_payload.get("machineType", "unknown")
        )
        expected["ACB_CLOUD_GPU_TYPE"] = accelerator_type or "<missing>"
        expected["ACB_CLOUD_GPU_COUNT"] = str(gpu_count) if gpu_count > 0 else "<missing>"
        expected["ACB_CLOUD_TTL_HOURS"] = (
            str(labels["ttl_hours"]) if labels.get("ttl_hours") else "<missing>"
        )
    for env_name, expected_value in expected.items():
        if expected_value and os.environ[env_name] != str(expected_value):
            errors.append(
                f"{env_name}={os.environ[env_name]} does not match preflight value {expected_value}"
            )
    warnings.append("cloud provenance environment is present")


def _doctor_config(
    config: BenchmarkConfig,
    *,
    runner: RunnerName,
    children: list[SweepChild],
    options: SweepOptions,
    check_gcloud: bool,
    gcloud_instance: str | None,
    gcloud_project: str | None,
    gcloud_zone: str | None,
    check_gcloud_quota: bool,
    check_local_port: bool,
    local_port: int,
    require_cloud_provenance: bool,
) -> DoctorResult:
    warnings: list[str] = []
    errors: list[str] = []
    try:
        budget = validate_budget(children, options)
    except ValueError as exc:
        budget = {
            "planned_runs": len(children),
            "planned_requests": sum(int(child.config.workload.request_count) for child in children),
            "estimated_gpu_hours": None,
        }
        if options.estimated_seconds_per_run is not None:
            budget["estimated_gpu_hours"] = (
                len(children) * options.estimated_seconds_per_run / 3600.0
            )
        errors.append(str(exc))

    backend_kinds = {child.config.backend.kind for child in children}
    if backend_kinds & REMOTE_BACKENDS:
        for child in children:
            backend = child.config.backend
            if backend.kind not in REMOTE_BACKENDS:
                continue
            if not backend.base_url.startswith(("http://", "https://")):
                errors.append(f"{child.config.run_name}: backend.base_url must be http(s)")
            if backend.kind == "vllm" and not backend.stream:
                warnings.append(f"{child.config.run_name}: streaming is off, TTFT is a proxy")
            if backend.kind == "vllm" and backend.scrape_metrics and not backend.metrics_url:
                errors.append(f"{child.config.run_name}: scrape_metrics requires metrics_url")
            if backend.kind == "vllm" and not backend.server_reset_command and len(children) > 1:
                warnings.append(
                    f"{child.config.run_name}: no server_reset_command for multi-run vLLM sweep"
                )
            if backend.kind == "vllm" and backend.adapter_model_names:
                missing = sorted(
                    adapter_id
                    for adapter_id in child.config.adapters.adapter_ids
                    if adapter_id not in backend.adapter_model_names
                )
                if missing:
                    errors.append(
                        f"{child.config.run_name}: missing vLLM model name(s) for adapters "
                        + ", ".join(missing)
                    )
        if any(child.config.cache.condition != "warm" for child in children):
            warnings.append(
                "non-warm cache conditions on remote backends need server counters before "
                "mechanism claims"
            )
        if check_gcloud:
            _check_gcloud_config(warnings, errors)
            if gcloud_zone:
                warnings.append(f"gcloud preflight target zone: {gcloud_zone}")
        instance_payload = None
        if gcloud_instance:
            instance_payload = _check_gcloud_instance(
                warnings,
                errors,
                instance=gcloud_instance,
                project=gcloud_project,
                zone=gcloud_zone,
            )
        if check_gcloud_quota:
            _check_gcloud_quota(
                warnings,
                errors,
                project=gcloud_project,
                zone=gcloud_zone,
                instance_payload=instance_payload,
            )
        if check_local_port:
            _check_local_port(warnings, errors, port=local_port)
            _check_backend_url_ports(
                warnings,
                errors,
                children=children,
                local_port=local_port,
            )
        if require_cloud_provenance:
            _check_cloud_provenance(
                warnings,
                errors,
                gcloud_instance=gcloud_instance,
                gcloud_project=gcloud_project,
                gcloud_zone=gcloud_zone,
                instance_payload=instance_payload,
            )
        elif (
            not check_gcloud
            and not gcloud_instance
            and any(
                "gcloud" in str(child.config.backend.server_reset_command or "")
                for child in children
            )
        ):
            warnings.append("gcloud reset command detected; rerun with --check-gcloud")

    return DoctorResult(
        runner=runner,
        planned_runs=int(budget["planned_runs"]),
        planned_requests=int(budget["planned_requests"]),
        estimated_gpu_hours=budget["estimated_gpu_hours"],
        warnings=sorted(set(warnings)),
        errors=sorted(set(errors)),
    )


def _print_doctor_result(result: DoctorResult) -> None:
    print(f"status: {'error' if result.errors else 'ok'}")
    print(f"runner: {result.runner}")
    print(f"planned runs: {result.planned_runs}")
    print(f"planned requests: {result.planned_requests}")
    if result.estimated_gpu_hours is not None:
        print(f"estimated GPU hours: {result.estimated_gpu_hours:.3f}")
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"- {error}")


def _doctor_json_payload(result: DoctorResult) -> dict[str, object]:
    return {
        "status": "error" if result.errors else "ok",
        "runner": result.runner,
        "planned_runs": result.planned_runs,
        "planned_requests": result.planned_requests,
        "estimated_gpu_hours": result.estimated_gpu_hours,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def _print_doctor_json_result(result: DoctorResult) -> None:
    print(json.dumps(_doctor_json_payload(result), sort_keys=True))


def _has_sweep_options(args: argparse.Namespace) -> bool:
    for name in SWEEP_OPTION_NAMES:
        value = getattr(args, name)
        if isinstance(value, bool):
            if value:
                return True
        elif value is not None:
            return True
    return False


def _report_callback(
    config: BenchmarkConfig,
    args: argparse.Namespace,
) -> Callable[[], None] | None:
    if args.no_report:
        return None
    return lambda: generate_report(
        config.output_dir,
        report_path=args.report_path,
        tables_dir=args.tables_dir,
    )


def _run_matrix(config: BenchmarkConfig, args: argparse.Namespace) -> None:
    execute_sweep(
        config=config,
        sweep_name=args.sweep_name or f"{config.run_name}-matrix",
        children=expand_matrix_sweep(config),
        run_child=lambda child_config, run_id: run_workload(
            child_config,
            run_id=run_id,
            generate_report_artifacts=False,
        ),
        record_dimensions=record_sweep_dimensions,
        options=options_from_args(args),
        on_complete=_report_callback(config, args),
    )


def _run_concurrency_sweep(config: BenchmarkConfig, args: argparse.Namespace) -> None:
    execute_sweep(
        config=config,
        sweep_name=args.sweep_name or f"{config.run_name}-concurrency",
        children=expand_concurrency_sweep_children(config),
        run_child=lambda child_config, run_id: run_concurrent(
            child_config,
            run_id=run_id,
            generate_report_artifacts=False,
        ),
        record_dimensions=record_sweep_dimensions,
        options=options_from_args(args),
        on_complete=_report_callback(config, args),
    )


def _run_exhaustive_sweep(config: BenchmarkConfig, args: argparse.Namespace) -> None:
    children = [
        SweepChild(child_config, dimensions)
        for child_config, dimensions in expand_exhaustive_sweep(config)
    ]
    execute_sweep(
        config=config,
        sweep_name=args.sweep_name or f"{config.run_name}-exhaustive",
        children=children,
        run_child=lambda child_config, run_id: run_concurrent(
            child_config,
            run_id=run_id,
            generate_report_artifacts=False,
        ),
        record_dimensions=record_sweep_dimensions,
        options=options_from_args(args),
        on_complete=_report_callback(config, args),
    )


def run_command(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> int:
    config = load_config(args.config)
    try:
        runner = infer_runner(config) if args.runner == "auto" else args.runner
    except ValueError as exc:
        if parser is not None:
            parser.error(str(exc))
        raise

    if runner in SWEEP_RUNNERS and args.run_id is not None:
        message = "--run-id is only supported by workload and concurrent runners"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    if runner not in SWEEP_RUNNERS and _has_sweep_options(args):
        message = "sweep options are only supported by matrix and sweep runners"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if runner == "workload":
        run_dir = run_workload(
            config,
            run_id=args.run_id,
            report_path=args.report_path,
            tables_dir=args.tables_dir,
            generate_report_artifacts=not args.no_report,
        )
        print(run_dir)
        return 0
    if runner == "concurrent":
        run_dir = run_concurrent(
            config,
            run_id=args.run_id,
            report_path=args.report_path,
            tables_dir=args.tables_dir,
            generate_report_artifacts=not args.no_report,
        )
        print(run_dir)
        return 0
    if runner == "matrix":
        _run_matrix(config, args)
        return 0
    if runner == "concurrency-sweep":
        _run_concurrency_sweep(config, args)
        return 0
    if runner == "exhaustive-sweep":
        _run_exhaustive_sweep(config, args)
        return 0

    raise AssertionError(f"unhandled runner: {runner}")


def report_command(args: argparse.Namespace) -> int:
    report_path = generate_report(
        args.runs_dir,
        report_path=args.report_path,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
    )
    print(report_path)
    return 0


def bundle_command(args: argparse.Namespace) -> int:
    try:
        manifest_path = build_evidence_bundle(
            bundle_name=args.bundle_name,
            runs_dir=args.runs_dir,
            output_dir=args.output_dir,
            run_ids=args.run_ids,
            run_globs=args.run_globs,
            reports=args.reports,
            figures=args.figures,
            tables=args.tables,
            repo_dir=args.repo_dir,
            strict=args.strict,
        )
    except EvidenceBundleValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(manifest_path)
    return 0


def doctor_command(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> int:
    config = load_config(args.config)
    try:
        runner = infer_runner(config) if args.runner == "auto" else args.runner
    except ValueError as exc:
        if parser is not None:
            parser.error(str(exc))
        raise
    children = _runner_children(config, runner)
    result = _doctor_config(
        config,
        runner=runner,
        children=children,
        options=_doctor_sweep_options(args),
        check_gcloud=args.check_gcloud,
        gcloud_instance=args.gcloud_instance,
        gcloud_project=args.gcloud_project,
        gcloud_zone=args.gcloud_zone,
        check_gcloud_quota=args.check_gcloud_quota,
        check_local_port=args.check_local_port,
        local_port=args.local_port,
        require_cloud_provenance=args.require_cloud_provenance,
    )
    if args.json_output:
        _print_doctor_json_result(result)
    else:
        _print_doctor_result(result)
    return 1 if result.errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acb", description="Adapter Cache Bench CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a benchmark config through an existing benchmark runner.",
    )
    run_parser.add_argument("--config", required=True, nargs="+", help="Config path(s) to load.")
    run_parser.add_argument(
        "--runner",
        choices=RUNNER_CHOICES,
        default="auto",
        help="Runner to use. Auto infers from matrix keys and concurrency settings.",
    )
    run_parser.add_argument("--run-id", help="Run id for workload and concurrent runners.")
    run_parser.add_argument("--report-path", default="reports/adapter-cache-tradeoffs.md")
    run_parser.add_argument("--tables-dir", default="reports/tables")
    run_parser.add_argument("--no-report", action="store_true", help="Skip report generation.")
    add_sweep_arguments(run_parser)
    run_parser.set_defaults(func=lambda args: run_command(args, run_parser))

    report_parser = subparsers.add_parser("report", help="Generate the benchmark report.")
    report_parser.add_argument("--runs-dir", default="artifacts/runs")
    report_parser.add_argument("--report-path", default="reports/adapter-cache-tradeoffs.md")
    report_parser.add_argument("--tables-dir", default="reports/tables")
    report_parser.add_argument("--figures-dir", default="reports/figures")
    report_parser.set_defaults(func=report_command)

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Build an evidence bundle manifest for selected runs and artifacts.",
    )
    bundle_parser.add_argument("--bundle-name", default="latest")
    bundle_parser.add_argument("--runs-dir", default="artifacts/runs")
    bundle_parser.add_argument("--output-dir")
    bundle_parser.add_argument("--run", dest="run_ids", action="append", default=[])
    bundle_parser.add_argument("--run-glob", dest="run_globs", action="append", default=[])
    bundle_parser.add_argument("--report", dest="reports", action="append", default=[])
    bundle_parser.add_argument("--figure", dest="figures", action="append", default=[])
    bundle_parser.add_argument("--table", dest="tables", action="append", default=[])
    bundle_parser.add_argument("--repo-dir", default=".")
    bundle_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero after writing the manifest if selected evidence is incomplete.",
    )
    bundle_parser.set_defaults(func=bundle_command)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run non-invasive config, budget, and optional gcloud preflight checks.",
    )
    doctor_parser.add_argument("--config", required=True, nargs="+", help="Config path(s) to load.")
    doctor_parser.add_argument(
        "--runner",
        choices=RUNNER_CHOICES,
        default="auto",
        help="Runner to use. Auto infers from matrix keys and concurrency settings.",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print a single machine-readable JSON object.",
    )
    doctor_parser.add_argument("--max-runs", type=int)
    doctor_parser.add_argument("--max-requests", type=int)
    doctor_parser.add_argument("--estimated-seconds-per-run", type=float)
    doctor_parser.add_argument("--max-estimated-gpu-hours", type=float)
    doctor_parser.add_argument(
        "--check-gcloud",
        action="store_true",
        help="Check local gcloud auth, project, and zone without starting resources.",
    )
    doctor_parser.add_argument(
        "--gcloud-instance",
        help="Describe an existing GCP VM without starting it.",
    )
    doctor_parser.add_argument(
        "--gcloud-project",
        help="Project to use for --gcloud-instance. Defaults to gcloud config.",
    )
    doctor_parser.add_argument(
        "--gcloud-zone",
        help="Zone to use for --gcloud-instance. Defaults to gcloud config.",
    )
    doctor_parser.add_argument(
        "--check-gcloud-quota",
        action="store_true",
        help="Check regional and project-wide GPU quota headroom without starting resources.",
    )
    doctor_parser.add_argument(
        "--check-local-port",
        action="store_true",
        help="Check whether the local tunnel port is already bound.",
    )
    doctor_parser.add_argument(
        "--local-port",
        type=_port_number,
        default=8000,
        help="Local tunnel port to check when --check-local-port is set.",
    )
    doctor_parser.add_argument(
        "--require-cloud-provenance",
        action="store_true",
        help="Require ACB_CLOUD_* and ACB_VLLM_IMAGE metadata before a cloud evidence run.",
    )
    doctor_parser.set_defaults(func=lambda args: doctor_command(args, doctor_parser))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
