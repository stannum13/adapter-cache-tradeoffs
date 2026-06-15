from adapter_cache_bench.config import WorkloadConfig
from adapter_cache_bench.workloads.generator import generate_workload


def test_all_workloads_generate_requested_records():
    names = [
        "shared_doc_qa",
        "mixed_tasks_same_doc",
        "agent_session",
        "low_overlap_control",
        "controlled_overlap",
        "prompt_layout_ablation",
    ]
    for name in names:
        records = generate_workload(WorkloadConfig(name=name, request_count=6, document_tokens=24))
        assert len(records) == 6
        assert all(record.prompt for record in records)


def test_prompt_layout_ablation_has_both_layouts():
    records = generate_workload(WorkloadConfig(name="prompt_layout_ablation", request_count=8))
    assert {record.prompt_layout for record in records} == {
        "instruction_before_document",
        "document_before_instruction",
    }


def test_controlled_overlap_changes_shared_prefix_reuse():
    low = generate_workload(
        WorkloadConfig(
            name="controlled_overlap",
            request_count=4,
            document_tokens=20,
            shared_prefix_fraction=0.0,
        )
    )
    high = generate_workload(
        WorkloadConfig(
            name="controlled_overlap",
            request_count=4,
            document_tokens=20,
            shared_prefix_fraction=0.75,
        )
    )

    assert "shared_0 shared_1" not in low[0].prompt
    assert "shared_0 shared_1" in high[0].prompt
    assert high[0].prompt.split()[:10] == high[1].prompt.split()[:10]


def test_jsonl_eval_workload_loads_public_domain_fixture():
    records = generate_workload(
        WorkloadConfig(
            name="jsonl_eval",
            dataset_path="data/eval/public_domain_eval.jsonl",
            request_count=5,
        )
    )

    assert len(records) == 5
    assert {record.task_type for record in records} >= {"qa", "json", "summary", "code"}
    assert records[1].requires_json


def test_jsonl_eval_workload_loads_large_public_domain_fixture():
    records = generate_workload(
        WorkloadConfig(
            name="jsonl_eval",
            dataset_path="data/eval/public_domain_eval_large.jsonl",
            request_count=100,
        )
    )

    assert len(records) == 100
    assert len({record.shared_prefix_id for record in records}) >= 5


def test_jsonl_eval_workload_loads_xlarge_public_domain_fixture():
    records = generate_workload(
        WorkloadConfig(
            name="jsonl_eval",
            dataset_path="data/eval/public_domain_eval_xlarge.jsonl",
            request_count=500,
        )
    )

    assert len(records) == 500
    assert len({record.shared_prefix_id for record in records}) >= 25
