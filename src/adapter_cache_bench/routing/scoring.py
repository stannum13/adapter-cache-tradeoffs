from __future__ import annotations

QUALITY_PRIORS: dict[tuple[str, str], float] = {
    ("qa", "qa"): 0.92,
    ("json", "json"): 0.94,
    ("summary", "summary"): 0.90,
    ("code", "code"): 0.88,
    ("qa", "multitask"): 0.82,
    ("json", "multitask"): 0.80,
    ("summary", "multitask"): 0.82,
    ("code", "multitask"): 0.76,
}


def quality_prior(task_type: str, adapter_id: str) -> float:
    if (task_type, adapter_id) in QUALITY_PRIORS:
        return QUALITY_PRIORS[(task_type, adapter_id)]
    if adapter_id == task_type:
        return 0.86
    return 0.46


def expected_adapter_for_task(task_type: str) -> str:
    if task_type in {"qa", "json", "summary", "code"}:
        return task_type
    return "multitask"
