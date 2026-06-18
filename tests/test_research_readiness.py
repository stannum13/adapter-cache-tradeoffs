from adapter_cache_bench.analysis.research_readiness import (
    check_research_readiness,
    format_markdown,
)


def test_research_readiness_reports_all_lanes(tmp_path):
    items = check_research_readiness(runs_dir=tmp_path)
    by_name = {item.name: item for item in items}

    assert set(by_name) == {
        "external_eval_preflight",
        "independent_external_eval",
        "per_condition_vllm_cache_isolation",
        "multi_model_comparison",
        "capacity_frontier_evidence",
        "adapter_aware_serving_metrics",
        "public_result_refresh",
    }
    assert by_name["external_eval_preflight"].status == "ok"
    assert by_name["independent_external_eval"].status == "needs_evidence"
    assert by_name["per_condition_vllm_cache_isolation"].status == "ok"
    assert by_name["multi_model_comparison"].status == "needs_evidence"
    assert by_name["capacity_frontier_evidence"].status == "ok"


def test_research_readiness_formats_markdown(tmp_path):
    table = format_markdown(check_research_readiness(runs_dir=tmp_path))

    assert "| item | status | detail |" in table
    assert "external_eval_preflight" in table
    assert "independent_external_eval" in table


def test_multi_model_readiness_requires_observed_runs(tmp_path):
    model_config = tmp_path / "models.yaml"
    model_config.write_text(
        """
matrix:
  models:
    - name: family-a
      alias: a
      adapter_model_names:
        qa: a-qa
        json: a-json
        summary: a-summary
        code: a-code
        multitask: a-multitask
    - name: family-b
      alias: b
      adapter_model_names:
        qa: b-qa
        json: b-json
        summary: b-summary
        code: b-code
        multitask: b-multitask
""",
        encoding="utf-8",
    )

    items = check_research_readiness(runs_dir=tmp_path, model_family_config=model_config)
    by_name = {item.name: item for item in items}

    assert by_name["multi_model_comparison"].status == "needs_evidence"
