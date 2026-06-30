import json

import pandas as pd
import pytest

from adapter_cache_bench.analysis.policy_regret import (
    POLICY_REGRET_COLUMNS,
    build_policy_regret_table,
    write_policy_regret_table,
)


def write_run(
    runs_dir,
    run_id,
    *,
    router_policy="cache_aware",
    cache_model="activated_lora",
    strategy="specialists",
    seed=17,
    request_count=100,
    qag=3.0,
    workload="controlled_overlap",
    overlap_fraction=0.75,
    concurrency=8,
    tenants=4,
    include_manifest=True,
):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "router_policy": router_policy,
                "cache_model": cache_model,
                "workload": workload,
                "request_count": request_count,
                "quality_adjusted_goodput": qag,
            }
        ),
        encoding="utf-8",
    )
    if include_manifest:
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "sweep_dimensions": {
                        "strategy": strategy,
                        "router": router_policy,
                        "cache": cache_model,
                        "workload": workload,
                        "seed": seed,
                        "overlap_fraction": overlap_fraction,
                        "concurrency": concurrency,
                        "tenants": tenants,
                    }
                }
            ),
            encoding="utf-8",
        )


def test_policy_regret_uses_best_observed_oracle_baseline(tmp_path):
    write_run(
        tmp_path,
        "oracle",
        router_policy="oracle",
        strategy="oracle",
        cache_model="activated_lora",
        qag=10.0,
        request_count=50,
    )
    write_run(
        tmp_path,
        "cache-aware",
        router_policy="cache_aware",
        strategy="specialists",
        cache_model="activated_lora",
        qag=7.5,
        request_count=100,
    )
    write_run(
        tmp_path,
        "multitask",
        router_policy="multitask",
        strategy="multitask",
        cache_model="base_shared",
        qag=5.0,
        request_count=100,
    )

    table = build_policy_regret_table(tmp_path)

    assert list(table.columns) == [
        *POLICY_REGRET_COLUMNS,
        "sweep_concurrency",
        "sweep_overlap_fraction",
        "sweep_tenants",
    ]
    assert len(table) == 3
    specialist = table[table["policy"].eq("specialists")].iloc[0]
    assert specialist["best_qag"] == 10.0
    assert specialist["regret"] == 2.5
    assert specialist["relative_regret"] == pytest.approx(0.25)
    assert specialist["rank"] == 2
    assert specialist["baseline_policy"] == "oracle"
    assert specialist["baseline_source"] == "oracle"
    assert bool(specialist["oracle_present"]) is True


def test_policy_regret_falls_back_to_best_observed_when_oracle_missing(tmp_path):
    write_run(
        tmp_path,
        "specialists-seed17",
        strategy="specialists",
        seed=17,
        qag=6.0,
        request_count=40,
    )
    write_run(
        tmp_path,
        "specialists-seed23",
        strategy="specialists",
        seed=23,
        qag=8.0,
        request_count=60,
    )
    write_run(
        tmp_path,
        "multitask",
        router_policy="multitask",
        cache_model="base_shared",
        strategy="multitask",
        seed=17,
        qag=5.0,
        request_count=100,
    )

    table = build_policy_regret_table(tmp_path)

    assert len(table) == 2
    specialist = table[table["policy"].eq("specialists")].iloc[0]
    multitask = table[table["policy"].eq("multitask")].iloc[0]
    assert specialist["runs"] == 2
    assert specialist["requests"] == 100
    assert specialist["qag"] == 7.0
    assert specialist["best_qag"] == 7.0
    assert specialist["regret"] == 0.0
    assert specialist["rank"] == 1
    assert specialist["baseline_source"] == "best_observed"
    assert bool(specialist["oracle_present"]) is False
    assert multitask["regret"] == 2.0
    assert multitask["relative_regret"] == pytest.approx(2.0 / 7.0)


def test_policy_regret_keeps_non_policy_sweep_dimensions_comparable(tmp_path):
    write_run(
        tmp_path,
        "low-overlap-specialists",
        strategy="specialists",
        qag=8.0,
        overlap_fraction=0.25,
    )
    write_run(
        tmp_path,
        "low-overlap-multitask",
        router_policy="multitask",
        cache_model="base_shared",
        strategy="multitask",
        qag=6.0,
        overlap_fraction=0.25,
    )
    write_run(
        tmp_path,
        "high-overlap-specialists",
        strategy="specialists",
        qag=4.0,
        overlap_fraction=0.95,
    )
    write_run(
        tmp_path,
        "high-overlap-multitask",
        router_policy="multitask",
        cache_model="base_shared",
        strategy="multitask",
        qag=9.0,
        overlap_fraction=0.95,
    )

    table = build_policy_regret_table(tmp_path)

    assert table["regime_id"].nunique() == 2
    assert "sweep_seed" not in table.columns
    assert "sweep_router" not in table.columns
    assert "sweep_cache" not in table.columns
    assert "sweep_overlap_fraction" in table.columns
    low_overlap = table[table["sweep_overlap_fraction"].eq(0.25)]
    high_overlap = table[table["sweep_overlap_fraction"].eq(0.95)]
    assert low_overlap[low_overlap["policy"].eq("specialists")].iloc[0]["rank"] == 1
    assert high_overlap[high_overlap["policy"].eq("multitask")].iloc[0]["rank"] == 1


def test_policy_regret_keeps_cache_conditions_comparable(tmp_path):
    write_run(
        tmp_path,
        "warm-specialists",
        strategy="specialists",
        qag=8.0,
    )
    write_run(
        tmp_path,
        "warm-multitask",
        router_policy="multitask",
        cache_model="base_shared",
        strategy="multitask",
        qag=6.0,
    )
    write_run(
        tmp_path,
        "disabled-specialists",
        strategy="specialists",
        qag=5.0,
    )
    write_run(
        tmp_path,
        "disabled-multitask",
        router_policy="multitask",
        cache_model="base_shared",
        strategy="multitask",
        qag=7.0,
    )
    for run_id, condition in [
        ("warm-specialists", "warm"),
        ("warm-multitask", "warm"),
        ("disabled-specialists", "prefix_disabled"),
        ("disabled-multitask", "prefix_disabled"),
    ]:
        manifest_path = tmp_path / run_id / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["sweep_dimensions"]["cache_condition"] = condition
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    table = build_policy_regret_table(tmp_path)

    assert table["regime_id"].nunique() == 2
    assert "sweep_cache_condition" in table.columns
    warm = table[table["sweep_cache_condition"].eq("warm")]
    disabled = table[table["sweep_cache_condition"].eq("prefix_disabled")]
    assert warm[warm["policy"].eq("specialists")].iloc[0]["rank"] == 1
    assert disabled[disabled["policy"].eq("multitask")].iloc[0]["rank"] == 1


def test_policy_regret_uses_summary_cache_condition_without_sweep(tmp_path):
    write_run(
        tmp_path,
        "warm-specialists",
        strategy="specialists",
        qag=8.0,
        include_manifest=False,
    )
    write_run(
        tmp_path,
        "cold-specialists",
        strategy="specialists",
        qag=5.0,
        include_manifest=False,
    )
    for run_id, condition in [("warm-specialists", "warm"), ("cold-specialists", "cold")]:
        summary_path = tmp_path / run_id / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["cache_condition"] = condition
        summary_path.write_text(json.dumps(payload), encoding="utf-8")

    table = build_policy_regret_table(tmp_path)

    assert table["regime_id"].nunique() == 2
    assert "cache_condition" in table.columns


def test_policy_regret_backfills_missing_sweep_cache_condition_from_summary(tmp_path):
    write_run(
        tmp_path,
        "legacy-warm",
        strategy="specialists",
        qag=8.0,
        include_manifest=False,
    )
    write_run(
        tmp_path,
        "new-cold",
        strategy="specialists",
        qag=5.0,
    )
    legacy_summary = tmp_path / "legacy-warm" / "summary.json"
    legacy_payload = json.loads(legacy_summary.read_text(encoding="utf-8"))
    legacy_payload["cache_condition"] = "warm"
    legacy_summary.write_text(json.dumps(legacy_payload), encoding="utf-8")
    cold_summary = tmp_path / "new-cold" / "summary.json"
    cold_payload = json.loads(cold_summary.read_text(encoding="utf-8"))
    cold_payload["cache_condition"] = "cold"
    cold_summary.write_text(json.dumps(cold_payload), encoding="utf-8")
    cold_manifest = tmp_path / "new-cold" / "manifest.json"
    cold_manifest_payload = json.loads(cold_manifest.read_text(encoding="utf-8"))
    cold_manifest_payload["sweep_dimensions"]["cache_condition"] = "cold"
    cold_manifest.write_text(json.dumps(cold_manifest_payload), encoding="utf-8")

    table = build_policy_regret_table(tmp_path)

    assert set(table["sweep_cache_condition"]) == {"warm", "cold"}


def test_policy_regret_handles_missing_manifest_and_empty_csv(tmp_path):
    write_run(
        tmp_path,
        "no-manifest",
        include_manifest=False,
        strategy="specialists",
        qag=2.0,
        request_count=10,
    )

    table = build_policy_regret_table(tmp_path)

    assert len(table) == 1
    assert table.iloc[0]["regime_key"] == "workload=controlled_overlap"
    assert table.iloc[0]["requests"] == 10

    output = tmp_path / "nested" / "policy_regret.csv"
    path = write_policy_regret_table(tmp_path / "missing", output)
    assert path == output
    text = output.read_text(encoding="utf-8")
    assert text.startswith(",".join(POLICY_REGRET_COLUMNS))


def test_policy_regret_ignores_runs_without_qag(tmp_path):
    run_dir = tmp_path / "missing-qag"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "missing-qag",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "workload": "controlled_overlap",
                "request_count": 100,
            }
        ),
        encoding="utf-8",
    )

    table = build_policy_regret_table(tmp_path)

    assert isinstance(table, pd.DataFrame)
    assert table.empty
