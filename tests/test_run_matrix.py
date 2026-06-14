from specialization_cache_frontier.bench.run_matrix import expand_matrix
from specialization_cache_frontier.config import BenchmarkConfig


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
