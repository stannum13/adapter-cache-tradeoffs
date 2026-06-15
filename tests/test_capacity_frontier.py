from adapter_cache_bench.analysis.capacity_frontier import (
    capacity_summary,
    load_capacity_frontier,
    write_capacity_tables,
)


def test_capacity_frontier_records_l4_failure_and_h100_success():
    df = load_capacity_frontier("data/results/capacity_frontier.yaml")

    by_id = {row.condition_id: row for row in df.itertuples()}
    assert by_id["qwen7b-l4-10-loras"].status == "fails"
    assert by_id["qwen7b-h100-10-loras"].status == "starts"
    assert by_id["qwen7b-h100-10-loras"].gpu_kv_cache_tokens == 998768
    assert "No available memory" in by_id["qwen7b-l4-10-loras"].failure_message


def test_capacity_summary_has_public_columns():
    summary = capacity_summary(load_capacity_frontier("data/results/capacity_frontier.yaml"))

    assert list(summary.columns) == [
        "condition_id",
        "gpu_model",
        "gpu_memory_gb",
        "max_model_len",
        "lora_count",
        "status",
        "available_kv_cache_gib",
        "gpu_kv_cache_tokens",
        "max_concurrency_at_context",
        "failure_message",
    ]
    assert len(summary) == 4


def test_capacity_frontier_writes_csv(tmp_path):
    output = tmp_path / "capacity.csv"

    path = write_capacity_tables(
        input_path="data/results/capacity_frontier.yaml",
        output_csv=output,
    )

    assert path == output
    text = output.read_text(encoding="utf-8")
    assert "qwen7b-h100-10-loras" in text
    assert "qwen7b-l4-8-loras" in text
