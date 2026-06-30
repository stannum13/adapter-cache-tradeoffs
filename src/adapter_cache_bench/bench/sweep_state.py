from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapter_cache_bench.config import BenchmarkConfig

REQUIRED_COMPLETE_ARTIFACTS = (
    "requests.jsonl",
    "summary.json",
    "config_resolved.yaml",
    "manifest.json",
    "status.json",
)


@dataclass(frozen=True)
class SweepChild:
    config: BenchmarkConfig
    dimensions: dict[str, Any]


@dataclass(frozen=True)
class SweepOptions:
    resume: bool = False
    continue_on_error: bool = False
    dry_run: bool = False
    max_runs: int | None = None
    max_requests: int | None = None
    estimated_seconds_per_run: float | None = None
    max_estimated_gpu_hours: float | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def child_id(run_name: str, dimensions: dict[str, Any]) -> str:
    payload = json.dumps(
        {"run_name": run_name, "dimensions": dimensions},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sweep_dir(config: BenchmarkConfig, sweep_name: str) -> Path:
    return Path(config.output_dir) / "_sweeps" / sweep_name


def artifact_complete(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    if not all((path / name).exists() for name in REQUIRED_COMPLETE_ARTIFACTS):
        return False
    try:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return status.get("status") == "complete"


def record_sweep_dimensions(run_dir: Path, dimensions: dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    manifest["sweep_dimensions"] = dimensions
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_plan(
    config: BenchmarkConfig,
    sweep_name: str,
    children: list[SweepChild],
) -> dict[str, Any]:
    entries = []
    for index, child in enumerate(children):
        run_name = child.config.run_name
        entries.append(
            {
                "index": index,
                "child_id": child_id(run_name, child.dimensions),
                "run_name": run_name,
                "run_dir": str(Path(child.config.output_dir) / run_name),
                "dimensions": child.dimensions,
                "request_count": child.config.workload.request_count,
            }
        )
    return {
        "schema_version": 1,
        "sweep_name": sweep_name,
        "parent_run_name": config.run_name,
        "created_at_utc": utc_now(),
        "output_dir": config.output_dir,
        "child_count": len(entries),
        "planned_request_count": sum(int(entry["request_count"]) for entry in entries),
        "children": entries,
    }


def write_plan(config: BenchmarkConfig, sweep_name: str, children: list[SweepChild]) -> Path:
    out = sweep_dir(config, sweep_name)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sweep_plan.json"
    path.write_text(
        json.dumps(build_plan(config, sweep_name, children), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_status(config: BenchmarkConfig, sweep_name: str) -> dict[str, Any]:
    path = sweep_dir(config, sweep_name) / "sweep_status.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _previous_children_by_id(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    children = status.get("children", [])
    if not isinstance(children, list):
        return {}
    return {str(child.get("child_id")): child for child in children if isinstance(child, dict)}


def write_status(
    config: BenchmarkConfig,
    sweep_name: str,
    plan: dict[str, Any],
    child_records: list[dict[str, Any]],
    *,
    status: str,
    started_at_utc: str,
    error: BaseException | None = None,
) -> Path:
    counts = {
        "planned": len(child_records),
        "complete": 0,
        "skipped": 0,
        "failed": 0,
        "running": 0,
        "pending": 0,
    }
    for child in child_records:
        child_status = str(child.get("status", "pending"))
        if child_status in counts:
            counts[child_status] += 1
    payload: dict[str, Any] = {
        "schema_version": 1,
        "sweep_name": sweep_name,
        "status": status,
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now(),
        "elapsed_s": max(
            0.0,
            datetime.now(timezone.utc).timestamp()
            - datetime.fromisoformat(started_at_utc).timestamp(),
        ),
        "plan_path": str(sweep_dir(config, sweep_name) / "sweep_plan.json"),
        "child_count": plan["child_count"],
        "planned_request_count": plan["planned_request_count"],
        "counts": counts,
        "children": child_records,
    }
    if error is not None:
        payload["exception_type"] = type(error).__name__
        payload["exception_message"] = str(error)
    out = sweep_dir(config, sweep_name)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sweep_status.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_budget(children: list[SweepChild], options: SweepOptions) -> dict[str, Any]:
    planned_runs = len(children)
    planned_requests = sum(int(child.config.workload.request_count) for child in children)
    estimated_gpu_hours = None
    if options.estimated_seconds_per_run is not None:
        estimated_gpu_hours = planned_runs * options.estimated_seconds_per_run / 3600.0
    if options.max_runs is not None and planned_runs > options.max_runs:
        raise ValueError(f"planned run count {planned_runs} exceeds --max-runs {options.max_runs}")
    if options.max_requests is not None and planned_requests > options.max_requests:
        raise ValueError(
            f"planned request count {planned_requests} exceeds --max-requests "
            f"{options.max_requests}"
        )
    if (
        options.max_estimated_gpu_hours is not None
        and estimated_gpu_hours is not None
        and estimated_gpu_hours > options.max_estimated_gpu_hours
    ):
        raise ValueError(
            f"estimated GPU hours {estimated_gpu_hours:.3f} exceeds "
            f"--max-estimated-gpu-hours {options.max_estimated_gpu_hours:.3f}"
        )
    return {
        "planned_runs": planned_runs,
        "planned_requests": planned_requests,
        "estimated_gpu_hours": estimated_gpu_hours,
    }


def add_sweep_arguments(parser) -> None:
    parser.add_argument("--sweep-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--estimated-seconds-per-run", type=float)
    parser.add_argument("--max-estimated-gpu-hours", type=float)


def options_from_args(args) -> SweepOptions:
    return SweepOptions(
        resume=bool(args.resume),
        continue_on_error=bool(args.continue_on_error),
        dry_run=bool(args.dry_run),
        max_runs=args.max_runs,
        max_requests=args.max_requests,
        estimated_seconds_per_run=args.estimated_seconds_per_run,
        max_estimated_gpu_hours=args.max_estimated_gpu_hours,
    )


def _initial_child_records(
    plan: dict[str, Any],
    previous_status: dict[str, Any],
) -> list[dict[str, Any]]:
    previous = _previous_children_by_id(previous_status)
    records = []
    for child in plan["children"]:
        old = previous.get(child["child_id"], {})
        records.append(
            {
                **child,
                "status": "pending",
                "attempts": int(old.get("attempts", 0) or 0),
                "started_at_utc": old.get("started_at_utc"),
                "finished_at_utc": old.get("finished_at_utc"),
                "exception_type": old.get("exception_type"),
                "exception_message": old.get("exception_message"),
            }
        )
    return records


def execute_sweep(
    *,
    config: BenchmarkConfig,
    sweep_name: str,
    children: list[SweepChild],
    run_child: Callable[[BenchmarkConfig, str], Path],
    record_dimensions: Callable[[Path, dict[str, Any]], None],
    options: SweepOptions,
    on_complete: Callable[[], None] | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    plan_path = write_plan(config, sweep_name, children)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    budget = validate_budget(children, options)
    previous_status = load_status(config, sweep_name) if options.resume else {}
    records = _initial_child_records(plan, previous_status)
    write_status(
        config,
        sweep_name,
        plan,
        records,
        status="dry_run" if options.dry_run else "running",
        started_at_utc=started_at,
    )
    print(
        "sweep plan: "
        f"{budget['planned_runs']} runs, {budget['planned_requests']} requests"
        + (
            f", {budget['estimated_gpu_hours']:.3f} estimated GPU hours"
            if budget["estimated_gpu_hours"] is not None
            else ""
        )
    )
    if options.dry_run:
        return load_status(config, sweep_name)

    sweep_error: BaseException | None = None
    for record, child in zip(records, children, strict=True):
        run_dir = Path(record["run_dir"])
        if options.resume and artifact_complete(run_dir):
            record["status"] = "skipped"
            record["finished_at_utc"] = utc_now()
            write_status(
                config,
                sweep_name,
                plan,
                records,
                status="running",
                started_at_utc=started_at,
            )
            print(f"skip complete child: {record['run_name']}")
            continue
        record["status"] = "running"
        record["attempts"] = int(record.get("attempts", 0) or 0) + 1
        record["started_at_utc"] = utc_now()
        record["exception_type"] = None
        record["exception_message"] = None
        write_status(config, sweep_name, plan, records, status="running", started_at_utc=started_at)
        try:
            child_run_dir = run_child(child.config, str(record["run_name"]))
            record_dimensions(child_run_dir, child.dimensions)
            record["run_dir"] = str(child_run_dir)
            record["status"] = "complete"
            record["finished_at_utc"] = utc_now()
            print(child_run_dir)
        except Exception as exc:
            record_dimensions(run_dir, child.dimensions)
            record["status"] = "failed"
            record["finished_at_utc"] = utc_now()
            record["exception_type"] = type(exc).__name__
            record["exception_message"] = str(exc)
            write_status(
                config,
                sweep_name,
                plan,
                records,
                status="running",
                started_at_utc=started_at,
            )
            if not options.continue_on_error:
                sweep_error = exc
                break
    final_status = "failed" if sweep_error else "complete"
    if sweep_error is None and any(record["status"] == "failed" for record in records):
        final_status = "complete_with_failures"
    write_status(
        config,
        sweep_name,
        plan,
        records,
        status=final_status,
        started_at_utc=started_at,
        error=sweep_error,
    )
    if sweep_error is not None:
        raise sweep_error
    if on_complete is not None:
        on_complete()
    return load_status(config, sweep_name)
