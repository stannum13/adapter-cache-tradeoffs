from specialization_cache_frontier.config import WorkloadConfig
from specialization_cache_frontier.workloads.generator import generate_workload


def test_all_workloads_generate_requested_records():
    names = [
        "shared_doc_qa",
        "mixed_tasks_same_doc",
        "agent_session",
        "low_overlap_control",
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
