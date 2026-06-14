from specialization_cache_frontier.config import deep_merge, load_config


def test_deep_merge_preserves_nested_defaults_and_applies_overrides():
    merged = deep_merge(
        {"router": {"policy": "cache_aware", "alpha": 0.01}, "cache": {"model": "standard_lora"}},
        {"router": {"policy": "semantic"}},
    )

    assert merged["router"]["policy"] == "semantic"
    assert merged["router"]["alpha"] == 0.01
    assert merged["cache"]["model"] == "standard_lora"


def test_load_config_composes_multiple_yaml_files(tmp_path):
    base = tmp_path / "base.yaml"
    router = tmp_path / "router.yaml"
    cache = tmp_path / "cache.yaml"
    base.write_text(
        """
run_name: composed
router:
  policy: cache_aware
  alpha: 0.02
cache:
  model: standard_lora
workload:
  request_count: 5
""",
        encoding="utf-8",
    )
    router.write_text(
        """
router:
  policy: semantic
""",
        encoding="utf-8",
    )
    cache.write_text(
        """
cache:
  model: activated_lora
""",
        encoding="utf-8",
    )

    config = load_config([base, router, cache])

    assert config.run_name == "composed"
    assert config.router.policy == "semantic"
    assert config.router.alpha == 0.02
    assert config.cache.model == "activated_lora"
    assert config.workload.request_count == 5
