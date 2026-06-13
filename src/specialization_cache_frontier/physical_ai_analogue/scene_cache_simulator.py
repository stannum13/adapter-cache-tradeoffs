from __future__ import annotations

import random
from dataclasses import dataclass

from specialization_cache_frontier.physical_ai_analogue.skill_router import (
    SKILL_PRIORS,
    route_skill,
)
from specialization_cache_frontier.physical_ai_analogue.world_model_cache import WorldModelCache


@dataclass
class SceneResult:
    skill_id: str
    cached_scene_tokens: int
    action_latency_ms: float
    success_probability: float
    safety_violation_probability: float


def make_scene(step: int, static_tokens: int = 48, dynamic_tokens: int = 8) -> list[str]:
    static = [f"static_{i}" for i in range(static_tokens)]
    dynamic = [f"dynamic_{step}_{i}" for i in range(dynamic_tokens)]
    return static + dynamic


def simulate(
    steps: int = 20, include_skill_in_key: bool = True, seed: int = 0
) -> list[SceneResult]:
    rng = random.Random(seed)
    cache = WorldModelCache(include_skill_in_key=include_skill_in_key)
    skills = ["pick", "place", "inspect", "generalist"]
    tasks = ["pick", "place", "inspect"]
    results = []
    for step in range(steps):
        task = tasks[step % len(tasks)]
        scene = make_scene(step)
        bonuses = {skill: cache.estimate(skill, scene) for skill in skills}
        skill = route_skill(task, skills, bonuses)
        cached = bonuses[skill]
        uncached = len(scene) - cached
        latency = 15.0 + uncached * 0.6
        success = min(0.99, SKILL_PRIORS.get((task, skill), 0.4) + rng.uniform(-0.02, 0.02))
        safety = max(0.001, 0.08 - success * 0.06)
        cache.observe(skill, scene)
        results.append(SceneResult(skill, cached, latency, success, safety))
    return results


if __name__ == "__main__":
    for row in simulate():
        print(row)
