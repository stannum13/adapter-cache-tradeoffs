from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from adapter_cache_bench.analysis.evidence_bundle import build_evidence_bundle
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


def _run_text_command(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return 127, str(exc)
    output = "\n".join(
        part.strip() for part in [completed.stdout, completed.stderr] if part.strip()
    )
    return completed.returncode, output


def _check_gcloud_config(warnings: list[str], errors: list[str]) -> None:
    if shutil.which("gcloud") is None:
        errors.append("gcloud is not installed or not on PATH")
        return

    account_code, accounts = _run_text_command(
        ["gcloud", "auth", "list", "--format=value(account)"]
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


def _doctor_config(
    config: BenchmarkConfig,
    *,
    runner: RunnerName,
    children: list[SweepChild],
    options: SweepOptions,
    check_gcloud: bool,
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
        elif any(
            "gcloud" in str(child.config.backend.server_reset_command or "") for child in children
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
    )
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
    )
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
    doctor_parser.add_argument("--max-runs", type=int)
    doctor_parser.add_argument("--max-requests", type=int)
    doctor_parser.add_argument("--estimated-seconds-per-run", type=float)
    doctor_parser.add_argument("--max-estimated-gpu-hours", type=float)
    doctor_parser.add_argument(
        "--check-gcloud",
        action="store_true",
        help="Check local gcloud auth, project, and zone without starting resources.",
    )
    doctor_parser.set_defaults(func=lambda args: doctor_command(args, doctor_parser))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
