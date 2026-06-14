import pytest

from specialization_cache_frontier.workloads.validate_dataset import validate_workload_config


def test_validate_public_domain_eval_config():
    result = validate_workload_config("configs/benchmark/public_domain_eval.yaml")

    assert result["request_count"] == 5
    assert set(result["task_types"]) >= {"qa", "json", "summary", "code"}


def test_validate_large_public_domain_eval_config():
    result = validate_workload_config("configs/benchmark/public_domain_eval_large.yaml")

    assert result["request_count"] == 100
    assert set(result["prompt_layouts"]) == {
        "document_before_instruction",
        "instruction_before_document",
    }


def test_validate_workload_rejects_missing_ground_truth(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        '{"request_id":"bad","task_type":"qa","document":"doc","question":"q","expected_adapter":"qa"}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
workload:
  name: jsonl_eval
  dataset_path: {dataset}
  request_count: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing ground_truth"):
        validate_workload_config(config)
