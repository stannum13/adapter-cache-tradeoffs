import pytest

from adapter_cache_bench.config import deep_merge, load_config


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


def test_load_config_rejects_unknown_cache_condition(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
cache:
  condition: not-a-real-condition
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(path)


def test_vllm_example_config_loads_optional_backend():
    config = load_config("configs/benchmark/vllm_example.yaml")

    assert config.backend.kind == "vllm"
    assert config.workload.request_count == 5


def test_benchmark_v0_mock_config_freezes_core_dimensions():
    config = load_config("configs/benchmark/benchmark_v0_mock.yaml")

    assert config.run_name == "benchmark-v0-mock"
    assert config.backend.kind == "mock"
    assert config.workload.request_count == 96
    assert config.cache.block_size == 8
    assert config.matrix["routers"] == [
        "semantic",
        "multitask",
        "sticky_session",
        "cache_aware",
        "oracle",
    ]
    assert config.matrix["caches"] == [
        "standard_lora",
        "activated_lora",
        "copy_on_write",
    ]
    assert config.matrix["seeds"] == [17, 23, 31]
