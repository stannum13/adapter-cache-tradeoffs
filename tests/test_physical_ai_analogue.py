from specialization_cache_frontier.physical_ai_analogue.scene_cache_simulator import simulate


def test_physical_ai_simulator_models_cache_reuse_effect():
    fragmented = simulate(steps=9, include_skill_in_key=True)
    shared = simulate(steps=9, include_skill_in_key=False)

    fragmented_latency = sum(row.action_latency_ms for row in fragmented) / len(fragmented)
    shared_latency = sum(row.action_latency_ms for row in shared) / len(shared)

    assert shared_latency <= fragmented_latency
    assert all(0.0 <= row.success_probability <= 1.0 for row in shared)
    assert all(0.0 <= row.safety_violation_probability <= 1.0 for row in shared)
