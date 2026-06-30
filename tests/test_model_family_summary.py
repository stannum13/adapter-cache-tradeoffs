import json

from adapter_cache_bench.analysis.model_family import (
    MODEL_FAMILY_COLUMNS,
    build_model_family_summary,
    write_model_family_summary,
)


def write_run(
    runs_dir,
    run_id,
    *,
    model_alias,
    strategy,
    backend_model,
    request_count,
    quality,
    p95_ttft_ms,
    qag,
    slo,
    server_hit,
):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "backend_model": backend_model,
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
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sweep_dimensions": {
                    "model_alias": model_alias,
                    "strategy": strategy,
                }
            }
        ),
        encoding="utf-8",
    )


def test_model_family_summary_groups_observed_runs(tmp_path):
    write_run(
        tmp_path,
        "model-family-vllm-specialists-a-seed17",
        model_alias="a",
        strategy="specialists",
        backend_model="family-a",
        request_count=100,
        quality=0.5,
        p95_ttft_ms=90.0,
        qag=2.0,
        slo=1.0,
        server_hit=0.8,
    )
    write_run(
        tmp_path,
        "model-family-vllm-specialists-a-seed23",
        model_alias="a",
        strategy="specialists",
        backend_model="family-a",
        request_count=100,
        quality=0.7,
        p95_ttft_ms=110.0,
        qag=4.0,
        slo=0.9,
        server_hit=0.6,
    )
    write_run(
        tmp_path,
        "unrelated-run",
        model_alias="b",
        strategy="multitask",
        backend_model="family-b",
        request_count=100,
        quality=0.1,
        p95_ttft_ms=500.0,
        qag=0.1,
        slo=0.0,
        server_hit=0.0,
    )

    summary = build_model_family_summary(tmp_path)

    assert list(summary.columns) == MODEL_FAMILY_COLUMNS
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["model_alias"] == "a"
    assert row["strategy"] == "specialists"
    assert row["runs"] == 2
    assert row["requests"] == 200
    assert row["mean_quality"] == 0.6
    assert row["p95_ttft_ms"] == 100.0
    assert row["quality_adjusted_goodput"] == 3.0
    assert row["slo_attainment_rate"] == 0.95
    assert row["server_prefix_cache_hit_rate"] == 0.7


def test_model_family_summary_writes_empty_csv(tmp_path):
    output = tmp_path / "summary.csv"

    path = write_model_family_summary(tmp_path, output)

    assert path == output
    text = output.read_text(encoding="utf-8")
    assert text.startswith(",".join(MODEL_FAMILY_COLUMNS))
