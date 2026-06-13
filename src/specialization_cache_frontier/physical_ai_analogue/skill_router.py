from __future__ import annotations

SKILL_PRIORS = {
    ("pick", "pick"): 0.94,
    ("place", "place"): 0.91,
    ("inspect", "inspect"): 0.89,
    ("pick", "generalist"): 0.78,
    ("place", "generalist"): 0.78,
    ("inspect", "generalist"): 0.80,
}


def route_skill(task: str, skills: list[str], cache_bonus: dict[str, int] | None = None) -> str:
    cache_bonus = cache_bonus or {}
    best = None
    for skill in skills:
        score = SKILL_PRIORS.get((task, skill), 0.40) + 0.01 * cache_bonus.get(skill, 0)
        if best is None or score > best[0]:
            best = (score, skill)
    assert best is not None
    return best[1]
