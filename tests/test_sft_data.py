import json

from experimental.training.build_sft_data import build_sft_split


def test_build_sft_split_writes_task_files(tmp_path):
    paths = build_sft_split(
        ["configs/benchmark/public_domain_eval.yaml"],
        tmp_path,
        eval_fraction=0.2,
    )

    train_rows = [
        json.loads(line)
        for line in paths["train"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert train_rows
    assert {"prompt", "completion", "task_type", "adapter_id"} <= set(train_rows[0])
    assert paths["train_qa"].exists()
    assert paths["train_json"].exists()
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["row_count"] == 5
