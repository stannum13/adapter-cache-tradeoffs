import json

import pandas as pd
import pytest

from adapter_cache_bench.analysis.claim_tables import (
    CLAIM_EVIDENCE_COLUMNS,
    build_claim_evidence_table,
    write_claim_evidence_table,
)


def write_run(
    runs_dir,
    run_id,
    *,
    model_alias="qwen7b",
    strategy="specialists",
    seed=17,
    request_count=100,
    quality=0.8,
    p95_ttft_ms=100.0,
    qag=3.0,
    slo=0.9,
    server_hit=0.5,
    include_manifest=True,
):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "backend_model": f"example/{model_alias}",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "workload": "controlled_overlap",
                "request_count": request_count,
                "mean_quality": quality,
                "p95_ttft_ms": p95_ttft_ms,
                "quality_adjusted_goodput": qag,
                "slo_attainment_rate": slo,
                "server_prefix_cache_hit_rate": server_hit,
            }
        ),
        encoding="utf-8",
    )
    if include_manifest:
        adapter_count = 4 if strategy == "specialists" else 1
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "sweep_dimensions": {
                        "strategy": strategy,
                        "concurrency": 8,
                        "model_alias": model_alias,
                        "workload": "controlled_overlap",
                        "cache": "activated_lora",
                        "seed": seed,
                        "overlap_fraction": 0.75,
                        "adapter_count": adapter_count,
                        "tenants": 4,
                        "isolation_scope": "trust_group",
                    }
                }
            ),
            encoding="utf-8",
        )


def test_claim_evidence_groups_runs_and_adds_ci_for_repeats(tmp_path):
    write_run(
        tmp_path,
        "specialists-seed17",
        seed=17,
        quality=0.7,
        p95_ttft_ms=90.0,
        qag=2.0,
        slo=0.8,
        server_hit=0.4,
    )
    write_run(
        tmp_path,
        "specialists-seed23",
        seed=23,
        quality=0.9,
        p95_ttft_ms=110.0,
        qag=4.0,
        slo=1.0,
        server_hit=0.8,
    )

    table = build_claim_evidence_table(tmp_path)

    evidence = table[table["row_type"].eq("evidence")].iloc[0]
    assert evidence["model_alias"] == "qwen7b"
    assert evidence["strategy"] == "specialists"
    assert evidence["runs"] == 2
    assert evidence["requests"] == 200
    assert evidence["mean_quality"] == 0.8
    assert evidence["p95_ttft_ms"] == 100.0
    assert evidence["qag"] == 3.0
    assert evidence["server_prefix_cache_hit_rate"] == pytest.approx(0.6)
    assert pd.notna(evidence["qag_ci95_low"])
    assert evidence["qag_ci95_low"] < evidence["qag"] < evidence["qag_ci95_high"]
    assert "adapter_count=4" in evidence["claim_group"]


def test_claim_evidence_adds_paired_specialist_vs_multitask_deltas(tmp_path):
    write_run(tmp_path, "specialists-seed17", seed=17, qag=4.0, quality=0.8, p95_ttft_ms=80.0)
    write_run(
        tmp_path,
        "multitask-seed17",
        strategy="multitask",
        seed=17,
        qag=3.0,
        quality=0.7,
        p95_ttft_ms=100.0,
    )
    write_run(tmp_path, "specialists-seed23", seed=23, qag=6.0, quality=0.9, p95_ttft_ms=90.0)
    write_run(
        tmp_path,
        "multitask-seed23",
        strategy="multitask",
        seed=23,
        qag=5.0,
        quality=0.8,
        p95_ttft_ms=110.0,
    )

    table = build_claim_evidence_table(tmp_path)

    delta = table[table["row_type"].eq("paired_delta")].iloc[0]
    assert delta["strategy"] == "specialists_vs_multitask_delta"
    assert delta["paired_seed_count"] == 2
    assert delta["paired_baseline_strategy"] == "multitask"
    assert delta["paired_comparison_strategy"] == "specialists"
    assert delta["runs"] == 2
    assert delta["requests"] == 200
    assert delta["qag"] == 1.0
    assert delta["mean_quality"] == pytest.approx(0.1)
    assert delta["p95_ttft_ms"] == -20.0
    assert "adapter_count" not in delta["claim_group"]


def test_claim_evidence_handles_missing_manifest_and_empty_csv(tmp_path):
    write_run(
        tmp_path,
        "no-manifest",
        model_alias="family-a",
        include_manifest=False,
        request_count=10,
    )

    table = build_claim_evidence_table(tmp_path)

    assert list(table.columns) == CLAIM_EVIDENCE_COLUMNS
    assert len(table) == 1
    row = table.iloc[0]
    assert row["model_alias"] == "family-a"
    assert row["strategy"] == "activated_lora"
    assert row["runs"] == 1
    assert pd.isna(row["qag_ci95_low"])

    output = tmp_path / "empty" / "claim_evidence.csv"
    path = write_claim_evidence_table(tmp_path / "missing", output)
    assert path == output
    assert output.read_text(encoding="utf-8").startswith(",".join(CLAIM_EVIDENCE_COLUMNS))
