import json

from specialization_cache_frontier.bench.quality import evaluate_prediction, weighted_task_quality
from specialization_cache_frontier.types import QualityResult


def test_json_quality_scores_valid_schema_and_fields():
    result = evaluate_prediction(
        "json",
        "json",
        json.dumps({"document_id": 1, "field": "fact_1_2"}),
        {"document_id": 1, "field": "fact_1_2"},
    )

    assert result.valid_json_rate == 1.0
    assert result.schema_match == 1.0
    assert result.field_f1 == 1.0
    assert result.score == 1.0


def test_json_quality_penalizes_invalid_json():
    result = evaluate_prediction("json", "json", "{invalid", {"field": "x"})

    assert result.valid_json_rate == 0.0
    assert result.score == 0.0


def test_task_quality_metrics_are_task_specific():
    qa = evaluate_prediction("qa", "qa", "fact_1_2", "fact_1_2")
    code = evaluate_prediction(
        "code",
        "code",
        "parses_fact_ids",
        {"tests": ["parses_fact_ids", "handles_empty_lines"]},
    )
    summary = evaluate_prediction("summary", "summary", "Summary for document 1", "document 1")

    assert qa.exact_match_like_score == 1.0
    assert code.unit_test_like_score == 0.5
    assert summary.rubric_score is not None


def test_weighted_task_quality_handles_empty_and_nonempty_results():
    assert weighted_task_quality([]) == 0.0
    assert (
        weighted_task_quality(
            [
                QualityResult(task_type="qa", adapter_id="qa", score=1.0),
                QualityResult(task_type="json", adapter_id="json", score=0.5),
            ]
        )
        == 0.75
    )
