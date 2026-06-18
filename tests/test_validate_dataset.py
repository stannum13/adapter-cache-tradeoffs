import pytest

from adapter_cache_bench.workloads.validate_dataset import validate_workload_config


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


def test_validate_xlarge_public_domain_eval_config():
    result = validate_workload_config(
        "configs/benchmark/public_domain_eval_xlarge.yaml",
        min_records=500,
        required_tasks={"qa", "json", "summary", "code"},
        required_layouts={"document_before_instruction", "instruction_before_document"},
        balanced_tasks=True,
        min_shared_prefix_groups=4,
        require_tenant_fields=True,
    )

    assert result["request_count"] == 500
    assert set(result["task_types"]) == {"qa", "json", "summary", "code"}
    assert result["task_counts"] == {"code": 125, "json": 125, "qa": 125, "summary": 125}
    assert result["tenant_count"] == 2
    assert result["trust_group_count"] == 2


def test_validate_source_eval_config():
    result = validate_workload_config("configs/benchmark/source_eval.yaml")

    assert result["request_count"] == 24
    assert set(result["task_types"]) == {"qa", "json", "summary", "code"}
    assert set(result["prompt_layouts"]) == {
        "document_before_instruction",
        "instruction_before_document",
    }


def test_validate_expanded_source_eval_config_requires_provenance():
    result = validate_workload_config(
        "configs/benchmark/source_eval_expanded.yaml",
        min_records=200,
        required_tasks={"qa", "json", "summary", "code"},
        required_layouts={"document_before_instruction", "instruction_before_document"},
        balanced_tasks=True,
        min_shared_prefix_groups=15,
        require_tenant_fields=True,
        require_source_fields=True,
        require_public_domain_license=True,
    )

    assert result["request_count"] == 240
    assert result["task_counts"] == {"code": 60, "json": 60, "qa": 60, "summary": 60}
    assert result["repeated_shared_prefix_groups"] == 15
    assert result["tenant_count"] == 3


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


def test_validate_workload_rejects_too_few_records():
    with pytest.raises(ValueError, match="Expected at least 500 records"):
        validate_workload_config("configs/benchmark/source_eval.yaml", min_records=500)


def test_validate_workload_rejects_missing_required_layout(tmp_path):
    dataset = tmp_path / "one_layout.jsonl"
    dataset.write_text(
        (
            '{"request_id":"r1","document_id":"d1","task_type":"qa","document":"doc",'
            '"question":"q","ground_truth":"a","expected_adapter":"qa",'
            '"prompt_layout":"document_before_instruction"}\n'
            '{"request_id":"r2","document_id":"d1","task_type":"json","document":"doc",'
            '"question":"q","ground_truth":{},"expected_adapter":"json",'
            '"prompt_layout":"document_before_instruction"}\n'
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
workload:
  name: jsonl_eval
  dataset_path: {dataset}
  request_count: 2
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required prompt layouts"):
        validate_workload_config(
            config,
            required_layouts={"document_before_instruction", "instruction_before_document"},
        )


def test_validate_workload_rejects_missing_source_fields(tmp_path):
    dataset = tmp_path / "missing_source.jsonl"
    dataset.write_text(
        (
            '{"request_id":"r1","document_id":"d1","task_type":"qa","document":"doc",'
            '"question":"q","ground_truth":"a","expected_adapter":"qa",'
            '"prompt_layout":"document_before_instruction"}\n'
        ),
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

    with pytest.raises(ValueError, match="missing source provenance"):
        validate_workload_config(config, require_source_fields=True)


def test_validate_workload_accepts_non_public_domain_source_license(tmp_path):
    dataset = tmp_path / "cc_by.jsonl"
    dataset.write_text(
        (
            '{"request_id":"r1","document_id":"d1","shared_prefix_id":"d1",'
            '"task_type":"qa","document":"doc","question":"q","ground_truth":"a",'
            '"expected_adapter":"qa","prompt_layout":"document_before_instruction",'
            '"source_title":"Unit Source","source_url":"https://example.test/source",'
            '"source_license":"cc-by-4.0"}\n'
        ),
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

    result = validate_workload_config(
        config,
        require_source_fields=True,
        min_shared_prefix_groups=0,
    )

    assert result["request_count"] == 1


def test_validate_workload_can_require_public_domain_license(tmp_path):
    dataset = tmp_path / "cc_by.jsonl"
    dataset.write_text(
        (
            '{"request_id":"r1","document_id":"d1","shared_prefix_id":"d1",'
            '"task_type":"qa","document":"doc","question":"q","ground_truth":"a",'
            '"expected_adapter":"qa","prompt_layout":"document_before_instruction",'
            '"source_title":"Unit Source","source_url":"https://example.test/source",'
            '"source_license":"cc-by-4.0"}\n'
        ),
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

    with pytest.raises(ValueError, match="without public-domain license"):
        validate_workload_config(
            config,
            require_source_fields=True,
            require_public_domain_license=True,
            min_shared_prefix_groups=0,
        )
