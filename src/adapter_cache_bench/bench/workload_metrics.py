from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from adapter_cache_bench.types import RequestRecord, WorkloadStructureMetrics


def _entropy(labels: Iterable[str]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _gini_concentration(labels: Iterable[str]) -> float:
    counts = sorted(Counter(labels).values())
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    weighted_sum = sum((index + 1) * count for index, count in enumerate(counts))
    return (2 * weighted_sum) / (len(counts) * total) - (len(counts) + 1) / len(counts)


def _switch_rate(labels: list[str]) -> float:
    if len(labels) < 2:
        return 0.0
    switches = sum(
        1 for previous, current in zip(labels, labels[1:], strict=False) if previous != current
    )
    return switches / (len(labels) - 1)


def _mean_reuse_distance(labels: list[str]) -> float:
    previous_index: dict[str, int] = {}
    distances: list[int] = []
    for index, label in enumerate(labels):
        if label in previous_index:
            distances.append(index - previous_index[label] - 1)
        previous_index[label] = index
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _shared_prefix_reuse_ratio(requests: list[RequestRecord]) -> float:
    seen: set[str] = set()
    reusable = 0
    prefix_count = 0
    for request in requests:
        if request.shared_prefix_id is None:
            continue
        prefix_count += 1
        if request.shared_prefix_id in seen:
            reusable += 1
        seen.add(request.shared_prefix_id)
    if prefix_count == 0:
        return 0.0
    return reusable / prefix_count


def _session_locality(requests: list[RequestRecord]) -> float:
    if len(requests) < 2:
        return 0.0
    same_session_transitions = sum(
        1
        for previous, current in zip(requests, requests[1:], strict=False)
        if previous.session_id == current.session_id
    )
    return same_session_transitions / (len(requests) - 1)


def compute_workload_structure_metrics(
    requests: list[RequestRecord],
) -> WorkloadStructureMetrics:
    """Compute deterministic workload-shape metrics from request order.

    Entropy is Shannon entropy in bits. Gini concentration is the standard Gini
    coefficient over observed category counts. Reuse distance is the mean number
    of intervening requests between repeated expected-adapter uses.
    """

    adapter_labels = [request.expected_adapter for request in requests]
    task_labels = [request.task_type for request in requests]
    return WorkloadStructureMetrics(
        adapter_entropy=_entropy(adapter_labels),
        task_entropy=_entropy(task_labels),
        adapter_gini_concentration=_gini_concentration(adapter_labels),
        task_gini_concentration=_gini_concentration(task_labels),
        adapter_switch_rate=_switch_rate(adapter_labels),
        mean_reuse_distance=_mean_reuse_distance(adapter_labels),
        shared_prefix_reuse_ratio=_shared_prefix_reuse_ratio(requests),
        session_locality=_session_locality(requests),
    )
