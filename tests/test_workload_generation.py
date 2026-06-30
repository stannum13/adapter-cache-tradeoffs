from collections import Counter

from adapter_cache_bench.config import WorkloadConfig, load_config
from adapter_cache_bench.workloads.generator import generate_workload

REGIME_WORKLOADS = [
    "regime_uniform",
    "regime_zipfian",
    "regime_bursty_session",
    "regime_phase_shift",
    "regime_adversarial_churn",
]


def test_all_workloads_generate_requested_records():
    names = [
        "shared_doc_qa",
        "mixed_tasks_same_doc",
        "agent_session",
        "low_overlap_control",
        "controlled_overlap",
        "prompt_layout_ablation",
        *REGIME_WORKLOADS,
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


def test_jsonl_eval_workload_loads_expanded_source_fixture():
    records = generate_workload(
        WorkloadConfig(
            name="jsonl_eval",
            dataset_path="data/eval/source_eval_expanded.jsonl",
            request_count=240,
        )
    )

    assert len(records) == 240
    assert {record.task_type for record in records} == {"qa", "json", "summary", "code"}
    assert {record.prompt_layout for record in records} == {
        "document_before_instruction",
        "instruction_before_document",
    }
    assert len({record.shared_prefix_id for record in records}) == 15


def test_jsonl_eval_workload_loads_external_source_fixture():
    records = generate_workload(
        WorkloadConfig(
            name="jsonl_eval",
            dataset_path="data/eval/external_public_domain_eval.jsonl",
            request_count=500,
        )
    )

    assert len(records) == 500
    assert {record.task_type for record in records} == {"qa", "json", "summary", "code"}
    assert {record.prompt_layout for record in records} == {
        "document_before_instruction",
        "instruction_before_document",
    }
    assert len({record.shared_prefix_id for record in records}) == 25


def test_regime_workloads_are_stable_for_same_seed_and_change_by_seed():
    for name in REGIME_WORKLOADS:
        config = WorkloadConfig(
            name=name,
            request_count=32,
            shared_document_count=6,
            sessions=6,
            seed=17,
            document_tokens=36,
        )
        same_config = config.model_copy()
        different_seed = config.model_copy(update={"seed": 23})

        first = generate_workload(config)
        second = generate_workload(same_config)
        changed = generate_workload(different_seed)

        first_signature = [
            (record.task_type, record.session_id, record.shared_prefix_id, record.prompt)
            for record in first
        ]
        assert first_signature == [
            (record.task_type, record.session_id, record.shared_prefix_id, record.prompt)
            for record in second
        ]
        assert first_signature != [
            (record.task_type, record.session_id, record.shared_prefix_id, record.prompt)
            for record in changed
        ]


def test_regime_uniform_balances_tasks_and_prefixes():
    records = generate_workload(
        WorkloadConfig(
            name="regime_uniform",
            request_count=40,
            shared_document_count=5,
            sessions=8,
            seed=17,
            document_tokens=36,
        )
    )

    task_counts = Counter(record.task_type for record in records)
    prefix_counts = Counter(record.shared_prefix_id for record in records)

    assert set(task_counts) == {"qa", "json", "summary", "code"}
    assert max(task_counts.values()) - min(task_counts.values()) <= 1
    assert len(prefix_counts) == 5
    assert max(prefix_counts.values()) - min(prefix_counts.values()) <= 1


def test_regime_zipfian_skews_adapter_and_prefix_reuse():
    records = generate_workload(
        WorkloadConfig(
            name="regime_zipfian",
            request_count=88,
            shared_document_count=5,
            sessions=8,
            seed=17,
            document_tokens=36,
        )
    )

    task_counts = Counter(record.task_type for record in records)
    prefix_counts = Counter(record.shared_prefix_id for record in records)

    assert task_counts["qa"] > task_counts["json"] > task_counts["summary"] > task_counts["code"]
    assert prefix_counts["doc-0"] == max(prefix_counts.values())
    assert len({record.session_id for record in records}) <= 3


def test_regime_bursty_session_has_consecutive_session_locality():
    records = generate_workload(
        WorkloadConfig(
            name="regime_bursty_session",
            request_count=36,
            shared_document_count=4,
            sessions=6,
            seed=17,
            document_tokens=36,
        )
    )

    same_session_edges = sum(
        left.session_id == right.session_id
        for left, right in zip(records, records[1:], strict=False)
    )
    longest_run = 1
    current_run = 1
    for left, right in zip(records, records[1:], strict=False):
        if left.session_id == right.session_id:
            current_run += 1
        else:
            longest_run = max(longest_run, current_run)
            current_run = 1
    longest_run = max(longest_run, current_run)

    assert same_session_edges >= 24
    assert longest_run >= 4
    assert len({record.task_type for record in records}) == 4


def test_regime_phase_shift_changes_task_mix_and_prefix_set():
    records = generate_workload(
        WorkloadConfig(
            name="regime_phase_shift",
            request_count=44,
            shared_document_count=6,
            sessions=6,
            seed=17,
            document_tokens=36,
        )
    )
    first_half = records[:22]
    second_half = records[22:]

    first_tasks = Counter(record.task_type for record in first_half)
    second_tasks = Counter(record.task_type for record in second_half)
    first_prefixes = {record.shared_prefix_id for record in first_half}
    second_prefixes = {record.shared_prefix_id for record in second_half}

    assert first_tasks["qa"] + first_tasks["json"] > first_tasks["summary"] + first_tasks["code"]
    assert (
        second_tasks["summary"] + second_tasks["code"] > second_tasks["qa"] + second_tasks["json"]
    )
    assert first_prefixes.isdisjoint(second_prefixes)


def test_regime_adversarial_churn_rotates_adapters_sessions_and_prefixes():
    records = generate_workload(
        WorkloadConfig(
            name="regime_adversarial_churn",
            request_count=24,
            shared_document_count=4,
            sessions=6,
            seed=17,
            document_tokens=36,
        )
    )

    assert all(
        left.task_type != right.task_type for left, right in zip(records, records[1:], strict=False)
    )
    assert all(
        left.session_id != right.session_id
        for left, right in zip(records, records[1:], strict=False)
    )
    assert len({record.shared_prefix_id for record in records}) == 24
    assert {record.prompt_layout for record in records} == {
        "document_before_instruction",
        "instruction_before_document",
    }


def test_regime_v0_mock_config_loads_all_regime_workloads():
    config = load_config("configs/benchmark/regime_v0_mock.yaml")

    assert config.run_name == "regime-v0-mock"
    assert config.backend.kind == "mock"
    assert config.workload.name == "regime_uniform"
    assert config.matrix["workloads"] == REGIME_WORKLOADS
