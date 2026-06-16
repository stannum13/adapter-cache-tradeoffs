from adapter_cache_bench.workloads.build_public_domain_eval import (
    build_records,
    write_dataset,
)


def test_build_public_domain_eval_records_cover_tasks_and_layouts():
    records = build_records(20)

    assert len(records) == 20
    assert {record["task_type"] for record in records} == {"qa", "json", "summary", "code"}
    assert {record["prompt_layout"] for record in records} == {
        "document_before_instruction",
        "instruction_before_document",
    }
    assert all(record["ground_truth"] is not None for record in records)


def test_write_public_domain_eval_dataset(tmp_path):
    path = write_dataset(tmp_path / "eval.jsonl", count=7)

    assert path.exists()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 7
