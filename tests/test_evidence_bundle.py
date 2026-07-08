import hashlib
import json

import pytest

from adapter_cache_bench.analysis.evidence_bundle import (
    EvidenceBundleValidationError,
    build_evidence_bundle,
    sha256_file,
)


def _write_run(runs_dir, run_id: str, *, include_manifest: bool = True):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"run_id": run_id, "request_count": 1}),
        encoding="utf-8",
    )
    (run_dir / "config_resolved.yaml").write_text(
        "run_name: evidence-test\n",
        encoding="utf-8",
    )
    if include_manifest:
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "artifact_files": [
                        "requests.jsonl",
                        "summary.json",
                        "config_resolved.yaml",
                        "manifest.json",
                    ],
                }
            ),
            encoding="utf-8",
        )
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": run_id, "status": "complete"}),
        encoding="utf-8",
    )
    (run_dir / "requests.jsonl").write_text(
        json.dumps({"request": {"id": "r1"}}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_build_evidence_bundle_records_selected_runs_and_hashes(tmp_path):
    runs_dir = tmp_path / "artifacts" / "runs"
    run_a = _write_run(runs_dir, "run-a")
    _write_run(runs_dir, "run-b")
    report = tmp_path / "docs" / "release_report.md"
    report.parent.mkdir()
    report.write_text("# Release\n", encoding="utf-8")
    figure = tmp_path / "docs" / "figures" / "plot.png"
    figure.parent.mkdir()
    figure.write_bytes(b"png-ish")
    table = tmp_path / "reports" / "tables" / "claim_evidence.csv"
    table.parent.mkdir(parents=True)
    table.write_text("row_type,claim_group\n", encoding="utf-8")

    manifest_path = build_evidence_bundle(
        bundle_name="slice-b",
        runs_dir=runs_dir,
        output_dir=tmp_path / "evidence" / "slice-b",
        run_ids=["run-a"],
        reports=[report],
        figures=[figure],
        tables=[table],
        repo_dir=tmp_path,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["bundle_name"] == "slice-b"
    assert payload["run_count"] == 1
    assert payload["validation"] == {
        "complete": True,
        "missing_generated_artifact_count": 0,
        "missing_generated_artifacts": [],
        "missing_required_file_count": 0,
        "missing_required_files_by_run": [],
        "run_count": 1,
        "runs_missing_git_commit": 0,
        "runs_with_missing_required_files": 0,
        "status": "pass",
    }
    assert payload["raw_artifact_policy"]["raw_artifacts_copied"] is False
    assert "not copied" in payload["raw_artifact_policy"]["note"]
    assert payload["git"]["available"] is False

    run_record = payload["runs"][0]
    assert run_record["run_id"] == "run-a"
    assert run_record["presence"] == {
        "config_resolved_yaml": True,
        "manifest_json": True,
        "summary_json": True,
        "status_json": True,
    }
    assert run_record["missing_required_files"] == []
    assert run_record["run_git"] == {"commit": "abc123", "dirty": False}

    included = {item["role"]: item for item in run_record["included_files"]}
    assert included["summary"]["sha256"] == sha256_file(run_a / "summary.json")
    assert included["resolved_config"]["relative_path"] == "run-a/config_resolved.yaml"
    assert included["run_manifest"]["size_bytes"] > 0
    assert included["run_status"]["sha256"] == sha256_file(run_a / "status.json")
    assert all(item["exists"] for item in included.values())

    excluded = run_record["excluded_raw_artifacts"]
    assert [item["relative_path"] for item in excluded] == ["run-a/requests.jsonl"]
    assert excluded[0]["role"] == "raw_artifact"

    report_record = payload["generated_artifacts"]["reports"][0]
    figure_record = payload["generated_artifacts"]["figures"][0]
    table_record = payload["generated_artifacts"]["tables"][0]
    assert report_record["exists"] is True
    assert report_record["sha256"] == hashlib.sha256(b"# Release\n").hexdigest()
    assert figure_record["exists"] is True
    assert figure_record["sha256"] == hashlib.sha256(b"png-ish").hexdigest()
    assert table_record["role"] == "table"
    assert table_record["exists"] is True
    assert table_record["sha256"] == hashlib.sha256(b"row_type,claim_group\n").hexdigest()


def test_build_evidence_bundle_supports_globs_and_missing_manifest(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "alpha-1", include_manifest=False)
    _write_run(runs_dir, "beta-1")

    manifest_path = build_evidence_bundle(
        bundle_name="globbed",
        runs_dir=runs_dir,
        output_dir=tmp_path / "out",
        run_globs=["alpha-*"],
        repo_dir=tmp_path,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [run["run_id"] for run in payload["runs"]] == ["alpha-1"]
    run_record = payload["runs"][0]
    assert run_record["presence"]["manifest_json"] is False
    assert run_record["presence"]["status_json"] is True
    assert run_record["missing_required_files"] == ["alpha-1/manifest.json"]
    assert payload["validation"]["complete"] is False
    assert payload["validation"]["status"] == "incomplete"
    assert payload["validation"]["runs_with_missing_required_files"] == 1
    assert payload["validation"]["missing_required_file_count"] == 1
    assert payload["validation"]["missing_generated_artifact_count"] == 0
    assert payload["validation"]["missing_required_files_by_run"] == [
        {
            "run_id": "alpha-1",
            "missing_required_files": ["alpha-1/manifest.json"],
        }
    ]
    manifest_file = [
        item for item in run_record["included_files"] if item["role"] == "run_manifest"
    ][0]
    assert manifest_file["exists"] is False
    assert "sha256" not in manifest_file


def test_build_evidence_bundle_strict_raises_after_writing_manifest(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "alpha-1", include_manifest=False)

    with pytest.raises(EvidenceBundleValidationError) as exc_info:
        build_evidence_bundle(
            bundle_name="strict",
            runs_dir=runs_dir,
            output_dir=tmp_path / "out",
            run_ids=["alpha-1"],
            repo_dir=tmp_path,
            strict=True,
        )

    manifest_path = tmp_path / "out" / "bundle_manifest.json"
    assert exc_info.value.manifest_path == manifest_path
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["validation"]["complete"] is False
    assert exc_info.value.validation == payload["validation"]


def test_build_evidence_bundle_raises_for_missing_selected_run(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="run directory does not exist"):
        build_evidence_bundle(
            bundle_name="missing",
            runs_dir=runs_dir,
            output_dir=tmp_path / "out",
            run_ids=["does-not-exist"],
            repo_dir=tmp_path,
        )
