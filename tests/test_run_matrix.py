from adapter_cache_bench.bench.run_concurrency_sweep import expand_concurrency_sweep
from adapter_cache_bench.bench.run_matrix import expand_matrix
from adapter_cache_bench.config import BenchmarkConfig, load_config


def test_expand_matrix_supports_repeated_seed_dimension():
    config = BenchmarkConfig(
        matrix={
            "routers": ["semantic"],
            "caches": ["standard_lora", "activated_lora"],
            "workloads": ["shared_doc_qa"],
            "seeds": [11, 17],
        }
    )

    expanded = expand_matrix(config)

    assert len(expanded) == 4
    assert {child.workload.seed for child in expanded} == {11, 17}
    assert {child.backend.seed for child in expanded} == {11, 17}
    assert {child.router.seed for child in expanded} == {11, 17}
    assert all("seed" in child.run_name for child in expanded)


def test_memory_pressure_matrix_uses_finite_cache_budget():
    config = load_config("configs/benchmark/memory_pressure.yaml")
    expanded = expand_matrix(config)

    assert expanded
    assert all(child.cache.max_memory_tokens == 512 for child in expanded)
    assert {child.workload.seed for child in expanded} == {11, 17}


def test_expand_concurrency_sweep_applies_strategy_and_concurrency():
    config = BenchmarkConfig(
        run_name="frontier",
        matrix={
            "strategies": ["base", "specialists", "multitask"],
            "concurrencies": [1, 4],
            "seeds": [11],
        },
    )

    expanded = expand_concurrency_sweep(config)

    assert len(expanded) == 6
    assert {child.backend.max_concurrency for child in expanded} == {1, 4}
    assert any(child.router.policy == "multitask" for child in expanded)
    specialists = [child for child in expanded if "specialists" in child.run_name][0]
    assert specialists.backend.adapter_model_names["qa"] == "qa-lora"
    base = [child for child in expanded if "base" in child.run_name][0]
    assert base.backend.adapter_model_names == {}
