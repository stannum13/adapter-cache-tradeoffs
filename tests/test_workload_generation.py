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
