from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CORE_RUN_FILES = {
    "summary.json": "summary",
    "config_resolved.yaml": "resolved_config",
    "manifest.json": "run_manifest",
    "status.json": "run_status",
}
RAW_ARTIFACT_PATTERNS = (
    "requests.jsonl",
    "*.jsonl",
    "backend_metrics_*.prom",
    "backend_metrics_*_error.txt",
)
RAW_ARTIFACT_NOTE = (
    "Raw request logs, backend metric scrapes, and other heavyweight run artifacts are "
    "not copied into evidence bundles. They remain under the source run directory and "
    "are listed here only as excluded raw artifacts when present."
)


@dataclass(frozen=True)
class FileRecord:
    path: str
    relative_path: str
    role: str
    exists: bool
    sha256: str | None = None
    size_bytes: int | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_record(path: Path, *, role: str, relative_to: Path) -> FileRecord:
    try:
        relative_path = path.relative_to(relative_to)
    except ValueError:
        relative_path = path
    if not path.exists():
        return FileRecord(
            path=str(path),
            relative_path=str(relative_path),
            role=role,
            exists=False,
        )
    return FileRecord(
        path=str(path),
        relative_path=str(relative_path),
        role=role,
        exists=True,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _file_record_dict(record: FileRecord) -> dict[str, Any]:
    data: dict[str, Any] = {
        "path": record.path,
        "relative_path": record.relative_path,
        "role": record.role,
        "exists": record.exists,
    }
    if record.sha256 is not None:
        data["sha256"] = record.sha256
    if record.size_bytes is not None:
        data["size_bytes"] = record.size_bytes
    return data


def git_metadata(repo_dir: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo_dir)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "dirty": None}
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _is_raw_artifact(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in RAW_ARTIFACT_PATTERNS)


def _load_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _select_run_dirs(
    runs_dir: Path,
    *,
    run_ids: Iterable[str] | None = None,
    run_globs: Iterable[str] | None = None,
) -> list[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs directory does not exist: {runs_dir}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"runs path is not a directory: {runs_dir}")

    selected: dict[str, Path] = {}
    available_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    requested_ids = list(run_ids or [])
    requested_globs = list(run_globs or [])

    if not requested_ids and not requested_globs:
        return sorted(available_dirs, key=lambda path: path.name)

    for run_id in requested_ids:
        run_dir = runs_dir / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        selected[run_dir.name] = run_dir

    for pattern in requested_globs:
        for run_dir in available_dirs:
            if fnmatch.fnmatch(run_dir.name, pattern):
                selected[run_dir.name] = run_dir

    if not selected:
        raise ValueError(
            f"no run directories matched selectors under {runs_dir}: "
            f"run_ids={requested_ids}, run_globs={requested_globs}"
        )
    return sorted(selected.values(), key=lambda path: path.name)


def _run_record(run_dir: Path, runs_dir: Path) -> dict[str, Any]:
    included_files = [
        _file_record_dict(_path_record(run_dir / filename, role=role, relative_to=runs_dir))
        for filename, role in CORE_RUN_FILES.items()
    ]
    presence = {
        "summary_json": (run_dir / "summary.json").exists(),
        "config_resolved_yaml": (run_dir / "config_resolved.yaml").exists(),
        "manifest_json": (run_dir / "manifest.json").exists(),
        "status_json": (run_dir / "status.json").exists(),
    }
    excluded_raw = [
        _file_record_dict(_path_record(path, role="raw_artifact", relative_to=runs_dir))
        for path in sorted(run_dir.iterdir(), key=lambda candidate: candidate.name)
        if path.is_file() and _is_raw_artifact(path)
    ]
    missing_required = [
        item["relative_path"] for item in included_files if not bool(item.get("exists"))
    ]
    run_manifest = _load_run_manifest(run_dir)
    run_git = {
        "commit": run_manifest.get("git_commit"),
        "dirty": run_manifest.get("git_dirty"),
    }
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "presence": presence,
        "included_files": included_files,
        "missing_required_files": missing_required,
        "excluded_raw_artifacts": excluded_raw,
        "run_git": run_git,
    }


def _generated_records(paths: Iterable[str | Path], *, role: str) -> list[dict[str, Any]]:
    return [
        _file_record_dict(_path_record(Path(path), role=role, relative_to=Path(".")))
        for path in paths
    ]


def build_evidence_bundle(
    *,
    bundle_name: str,
    runs_dir: str | Path = "artifacts/runs",
    output_dir: str | Path | None = None,
    run_ids: Iterable[str] | None = None,
    run_globs: Iterable[str] | None = None,
    reports: Iterable[str | Path] | None = None,
    figures: Iterable[str | Path] | None = None,
    repo_dir: str | Path = ".",
) -> Path:
    runs_root = Path(runs_dir)
    bundle_dir = Path(output_dir) if output_dir is not None else Path("evidence") / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _select_run_dirs(runs_root, run_ids=run_ids, run_globs=run_globs)
    manifest = {
        "schema_version": 1,
        "bundle_name": bundle_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_runs_dir": str(runs_root),
        "output_dir": str(bundle_dir),
        "git": git_metadata(repo_dir),
        "raw_artifact_policy": {
            "raw_artifacts_copied": False,
            "note": RAW_ARTIFACT_NOTE,
        },
        "run_count": len(run_dirs),
        "runs": [_run_record(run_dir, runs_root) for run_dir in run_dirs],
        "generated_artifacts": {
            "reports": _generated_records(reports or [], role="report"),
            "figures": _generated_records(figures or [], role="figure"),
        },
    }
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a machine-readable evidence bundle manifest for benchmark runs."
    )
    parser.add_argument("--bundle-name", default="latest")
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output-dir")
    parser.add_argument("--run", dest="run_ids", action="append", default=[])
    parser.add_argument("--run-glob", dest="run_globs", action="append", default=[])
    parser.add_argument("--report", dest="reports", action="append", default=[])
    parser.add_argument("--figure", dest="figures", action="append", default=[])
    parser.add_argument("--repo-dir", default=".")
    args = parser.parse_args()

    manifest_path = build_evidence_bundle(
        bundle_name=args.bundle_name,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        run_ids=args.run_ids,
        run_globs=args.run_globs,
        reports=args.reports,
        figures=args.figures,
        repo_dir=args.repo_dir,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
