from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from specialization_cache_frontier.types import QualityResult


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def f1_score(predicted: set[str], expected: set[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match_like(prediction: str, ground_truth: Any) -> float:
    expected = str(ground_truth or "")
    if prediction.strip().lower() == expected.strip().lower():
        return 1.0
    return f1_score(_token_set(prediction), _token_set(expected))


def json_quality(prediction: str, ground_truth: Any) -> dict[str, float]:
    try:
        parsed = json.loads(prediction)
    except json.JSONDecodeError:
        return {"valid_json_rate": 0.0, "schema_match": 0.0, "field_f1": 0.0, "score": 0.0}
    valid_json_rate = 1.0
    if isinstance(ground_truth, Mapping):
        expected_keys = set(ground_truth.keys())
    else:
        expected_keys = set()
    parsed_mapping = parsed if isinstance(parsed, Mapping) else {}
    parsed_keys = set(parsed_mapping.keys())
    schema_match = (
        1.0 if expected_keys and expected_keys <= parsed_keys else float(not expected_keys)
    )
    expected_values = _token_set(json.dumps(ground_truth, sort_keys=True))
    parsed_values = _token_set(json.dumps(parsed, sort_keys=True))
    field_f1 = f1_score(parsed_values, expected_values)
    score = (valid_json_rate + schema_match + field_f1) / 3.0
    return {
        "valid_json_rate": valid_json_rate,
        "schema_match": schema_match,
        "field_f1": field_f1,
        "score": score,
    }


def code_quality(prediction: str, ground_truth: Any) -> float:
    tests = []
    if isinstance(ground_truth, Mapping):
        tests = list(ground_truth.get("tests", []))
    if not tests:
        return 0.5 if prediction.strip() else 0.0
    prediction_tokens = _token_set(prediction)
    passed = sum(1 for test in tests if _token_set(str(test)) <= prediction_tokens)
    return passed / len(tests)


def summary_quality(prediction: str, ground_truth: Any) -> float:
    prediction_tokens = _token_set(prediction)
    expected_tokens = _token_set(str(ground_truth or ""))
    length_bonus = 1.0 if 5 <= len(prediction_tokens) <= 80 else 0.7
    return min(1.0, f1_score(prediction_tokens, expected_tokens) * 0.8 + 0.2 * length_bonus)


def evaluate_prediction(
    task_type: str,
    adapter_id: str,
    prediction: str,
    ground_truth: Any,
) -> QualityResult:
    if task_type == "json":
        metrics = json_quality(prediction, ground_truth)
        return QualityResult(
            task_type=task_type,
            adapter_id=adapter_id,
            score=metrics["score"],
            valid_json_rate=metrics["valid_json_rate"],
            schema_match=metrics["schema_match"],
            field_f1=metrics["field_f1"],
        )
    if task_type == "code":
        score = code_quality(prediction, ground_truth)
        return QualityResult(
            task_type=task_type,
            adapter_id=adapter_id,
            score=score,
            unit_test_like_score=score,
        )
    if task_type == "summary":
        score = summary_quality(prediction, ground_truth)
        return QualityResult(
            task_type=task_type,
            adapter_id=adapter_id,
            score=score,
            rubric_score=score,
        )
    score = exact_match_like(prediction, ground_truth)
    return QualityResult(
        task_type=task_type,
        adapter_id=adapter_id,
        score=score,
        exact_match_like_score=score,
    )


def weighted_task_quality(results: list[QualityResult]) -> float:
    if not results:
        return 0.0
    return sum(result.score for result in results) / len(results)
