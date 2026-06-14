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
