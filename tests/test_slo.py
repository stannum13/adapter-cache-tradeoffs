import pandas as pd

from specialization_cache_frontier.analysis.slo import slo_sweep


def test_slo_sweep_computes_goodput_at_multiple_thresholds():
    request_df = pd.DataFrame(
        [
            {
                "run_id": "run",
                "workload": "shared_doc_qa",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "ttft_ms": 20.0,
                "e2e_ms": 100.0,
                "quality": 0.8,
            },
            {
                "run_id": "run",
                "workload": "shared_doc_qa",
                "router_policy": "cache_aware",
                "cache_model": "activated_lora",
                "ttft_ms": 80.0,
                "e2e_ms": 100.0,
                "quality": 1.0,
            },
        ]
    )

    table = slo_sweep(request_df, [50.0, 100.0])

    assert table.loc[table["ttft_slo_ms"].eq(50.0), "requests_under_slo"].iloc[0] == 1
    assert table.loc[table["ttft_slo_ms"].eq(100.0), "requests_under_slo"].iloc[0] == 2
    assert table.loc[table["ttft_slo_ms"].eq(100.0), "quality_adjusted_goodput"].iloc[0] == 9.0
