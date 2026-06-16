from adapter_cache_bench.analysis.research_readiness import (
    check_research_readiness,
    format_markdown,
)


def test_research_readiness_reports_all_lanes(tmp_path):
    items = check_research_readiness(runs_dir=tmp_path)
    by_name = {item.name: item for item in items}

    assert set(by_name) == {
        "external_eval_preflight",
        "per_condition_vllm_cache_isolation",
        "multi_model_comparison",
        "adapter_aware_serving_metrics",
        "public_result_refresh",
    }
    assert by_name["external_eval_preflight"].status == "ok"
    assert by_name["per_condition_vllm_cache_isolation"].status == "ok"
    assert by_name["multi_model_comparison"].status == "needs_evidence"


def test_research_readiness_formats_markdown(tmp_path):
    table = format_markdown(check_research_readiness(runs_dir=tmp_path))

    assert "| item | status | detail |" in table
    assert "external_eval_preflight" in table
