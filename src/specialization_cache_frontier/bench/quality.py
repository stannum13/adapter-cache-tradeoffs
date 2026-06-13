from __future__ import annotations

from specialization_cache_frontier.types import QualityResult


def weighted_task_quality(results: list[QualityResult]) -> float:
    if not results:
        return 0.0
    return sum(result.score for result in results) / len(results)
