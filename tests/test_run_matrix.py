from adapter_cache_bench.bench.run_concurrency_sweep import (
    expand_concurrency_sweep,
    expand_concurrency_sweep_children,
)
from adapter_cache_bench.bench.run_exhaustive_sweep import (
    expand_exhaustive_sweep,
    record_sweep_dimensions,
)
from adapter_cache_bench.bench.run_matrix import expand_matrix, expand_matrix_sweep
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


def test_expand_matrix_sweep_records_dimensions():
    config = BenchmarkConfig(
        matrix={
            "routers": ["semantic"],
            "caches": ["standard_lora"],
            "workloads": ["shared_doc_qa"],
            "seeds": [17],
        }
    )

    children = expand_matrix_sweep(config)

    assert len(children) == 1
    assert children[0].dimensions == {
        "router": "semantic",
        "cache": "standard_lora",
        "workload": "shared_doc_qa",
        "seed": 17,
    }


def test_expand_matrix_sweep_records_cache_condition_dimension_when_configured():
    config = BenchmarkConfig(
        matrix={
            "routers": ["semantic"],
            "caches": ["standard_lora"],
            "cache_conditions": ["warm", "prefix_disabled", "cold"],
            "workloads": ["shared_doc_qa"],
            "seeds": [17],
        }
    )

    children = expand_matrix_sweep(config)

    assert len(children) == 3
    assert {child.config.cache.condition for child in children} == {
        "warm",
        "prefix_disabled",
        "cold",
    }
    assert {child.dimensions["cache_condition"] for child in children} == {
        "warm",
        "prefix_disabled",
        "cold",
    }
    assert all("seed17" in child.config.run_name for child in children)


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


def test_expand_concurrency_sweep_applies_cache_conditions():
    config = BenchmarkConfig(
        run_name="frontier",
        matrix={
            "strategies": ["specialists"],
            "concurrencies": [4],
            "cache_conditions": ["warm", "prefix_disabled", "cold"],
            "seeds": [11],
        },
    )

    children = expand_concurrency_sweep_children(config)

    assert len(children) == 3
    assert {child.config.cache.condition for child in children} == {
        "warm",
        "prefix_disabled",
        "cold",
    }
    assert {child.dimensions["cache_condition"] for child in children} == {
        "warm",
        "prefix_disabled",
        "cold",
    }


def test_expand_exhaustive_sweep_applies_dimensions():
    config = BenchmarkConfig(
        run_name="exhaustive",
        matrix={
            "strategies": ["specialists"],
            "concurrencies": [8],
            "workloads": ["controlled_overlap"],
            "caches": ["activated_lora"],
            "cache_conditions": ["warm", "prefix_disabled"],
            "overlap_fractions": [0.25, 0.75],
            "adapter_counts": [2],
            "tenants": [1, 4],
            "isolation_scopes": ["tenant"],
            "seeds": [17],
        },
    )

    expanded = expand_exhaustive_sweep(config)

    assert len(expanded) == 8
    child, dimensions = expanded[0]
    assert child.backend.max_concurrency == 8
    assert child.workload.name == "controlled_overlap"
    assert child.workload.shared_prefix_fraction in {0.25, 0.75}
    assert child.workload.tenants in {1, 4}
    assert child.adapters.adapter_ids == ["qa", "json"]
    assert child.backend.adapter_model_names == {"qa": "qa-lora", "json": "json-lora"}
    assert dimensions["adapter_count"] == 2
    assert dimensions["isolation_scope"] == "tenant"
    assert dimensions["cache_condition"] in {"warm", "prefix_disabled"}
    assert child.cache.condition in {"warm", "prefix_disabled"}


def test_expand_exhaustive_sweep_supports_model_dimension():
    config = BenchmarkConfig(
        run_name="models",
        matrix={
            "strategies": ["specialists", "multitask"],
            "concurrencies": [8],
            "workloads": ["jsonl_eval"],
            "models": [
                {
                    "name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "alias": "qwen15b",
                    "adapter_model_names": {
                        "qa": "qwen15b-qa-lora",
                        "json": "qwen15b-json-lora",
                        "summary": "qwen15b-summary-lora",
                        "code": "qwen15b-code-lora",
                        "multitask": "qwen15b-multitask-lora",
                    },
                }
            ],
            "adapter_counts": [4],
            "seeds": [17],
        },
    )

    expanded = expand_exhaustive_sweep(config)

    specialists = [child for child, _ in expanded if "specialists" in child.run_name][0]
    multitask = [child for child, _ in expanded if "multitask" in child.run_name][0]
    assert specialists.backend.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert specialists.backend.adapter_model_names["qa"] == "qwen15b-qa-lora"
    assert multitask.backend.adapter_model_names == {"multitask": "qwen15b-multitask-lora"}
    assert all(dimensions["model_alias"] == "qwen15b" for _, dimensions in expanded)


def test_vllm_bridge_reset_config_expands_to_small_g8_bridge():
    config = load_config("configs/benchmark/vllm_bridge_reset.yaml")

    expanded = expand_exhaustive_sweep(config)

    assert len(expanded) == 12
    assert sum(child.workload.request_count for child, _ in expanded) == 288
    assert {dimensions["workload"] for _, dimensions in expanded} == {
        "controlled_overlap",
        "mixed_tasks_same_doc",
        "low_overlap_control",
    }
    assert {dimensions["strategy"] for _, dimensions in expanded} == {"specialists", "multitask"}
    assert {dimensions["cache_condition"] for _, dimensions in expanded} == {
        "warm",
        "prefix_disabled",
    }
    assert all(child.backend.kind == "vllm" for child, _ in expanded)
    assert all(child.backend.server_reset_command for child, _ in expanded)


def test_record_sweep_dimensions_updates_manifest(tmp_path):
    import json

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run"}), encoding="utf-8")

    record_sweep_dimensions(run_dir, {"strategy": "specialists", "concurrency": 8})

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sweep_dimensions"]["strategy"] == "specialists"
    assert manifest["sweep_dimensions"]["concurrency"] == 8
