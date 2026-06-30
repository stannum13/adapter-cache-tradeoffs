import pytest

from adapter_cache_bench.bench.workload_metrics import compute_workload_structure_metrics
from adapter_cache_bench.types import RequestRecord


def _request(
    index: int,
    *,
    task_type: str,
    expected_adapter: str,
    session_id: str,
    shared_prefix_id: str | None,
) -> RequestRecord:
    return RequestRecord(
        request_id=f"r{index}",
        session_id=session_id,
        tenant_id="tenant",
        trust_group_id="trust",
        task_type=task_type,
        prompt=f"prompt {index}",
        shared_prefix_id=shared_prefix_id,
        expected_adapter=expected_adapter,
    )


def test_workload_structure_metrics_are_deterministic_for_request_stream():
    requests = [
        _request(0, task_type="qa", expected_adapter="qa", session_id="s1", shared_prefix_id="d1"),
        _request(1, task_type="qa", expected_adapter="qa", session_id="s1", shared_prefix_id="d1"),
        _request(
            2, task_type="json", expected_adapter="json", session_id="s2", shared_prefix_id="d2"
        ),
        _request(3, task_type="qa", expected_adapter="qa", session_id="s2", shared_prefix_id="d1"),
    ]

    metrics = compute_workload_structure_metrics(requests)

    assert metrics.adapter_entropy == pytest.approx(0.811278, abs=1e-6)
    assert metrics.task_entropy == pytest.approx(0.811278, abs=1e-6)
    assert metrics.adapter_gini_concentration == pytest.approx(0.25)
    assert metrics.task_gini_concentration == pytest.approx(0.25)
    assert metrics.adapter_switch_rate == pytest.approx(2 / 3)
    assert metrics.mean_reuse_distance == pytest.approx(0.5)
    assert metrics.shared_prefix_reuse_ratio == pytest.approx(0.5)
    assert metrics.session_locality == pytest.approx(2 / 3)


def test_workload_structure_metrics_handle_empty_and_singleton_inputs():
    empty = compute_workload_structure_metrics([])
    assert empty.adapter_entropy == 0.0
    assert empty.task_entropy == 0.0
    assert empty.adapter_switch_rate == 0.0
    assert empty.mean_reuse_distance == 0.0
    assert empty.shared_prefix_reuse_ratio == 0.0
    assert empty.session_locality == 0.0

    singleton = compute_workload_structure_metrics(
        [
            _request(
                0,
                task_type="summary",
                expected_adapter="summary",
                session_id="s1",
                shared_prefix_id=None,
            )
        ]
    )
    assert singleton.adapter_entropy == 0.0
    assert singleton.task_entropy == 0.0
    assert singleton.adapter_gini_concentration == 0.0
    assert singleton.task_gini_concentration == 0.0
    assert singleton.adapter_switch_rate == 0.0
    assert singleton.mean_reuse_distance == 0.0
    assert singleton.shared_prefix_reuse_ratio == 0.0
    assert singleton.session_locality == 0.0
