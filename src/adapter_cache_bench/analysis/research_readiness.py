from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adapter_cache_bench.analysis.adapter_cache_metrics import build_adapter_cache_metrics
from adapter_cache_bench.analysis.capacity_frontier import load_capacity_frontier
from adapter_cache_bench.bench.aggregate import load_summaries
from adapter_cache_bench.config import load_config
from adapter_cache_bench.workloads.validate_dataset import validate_workload_config


@dataclass
class ReadinessItem:
    name: str
    status: str
    detail: str


def _ok(name: str, detail: str) -> ReadinessItem:
    return ReadinessItem(name=name, status="ok", detail=detail)


def _needs_evidence(name: str, detail: str) -> ReadinessItem:
    return ReadinessItem(name=name, status="needs_evidence", detail=detail)


def _missing(name: str, detail: str) -> ReadinessItem:
    return ReadinessItem(name=name, status="missing", detail=detail)


def _is_generated_external_fixture(dataset_path: str | None) -> bool:
    if not dataset_path:
        return True
    path = Path(dataset_path)
    return path.name.startswith("public_domain_eval")


def _complete_model_specs(model_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        spec
        for spec in model_specs
        if isinstance(spec, dict)
        and {"qa", "json", "summary", "code", "multitask"}
        <= set((spec.get("adapter_model_names") or {}).keys())
    ]


def _observed_model_family_aliases(runs_dir: str | Path) -> set[str]:
    df = load_summaries(runs_dir)
    if df.empty or "run_id" not in df:
        return set()
    model_family_rows = df[df["run_id"].str.contains("model-family-vllm", na=False)]
    if model_family_rows.empty or "sweep_model_alias" not in model_family_rows:
        return set()
    aliases = model_family_rows["sweep_model_alias"].dropna().astype(str)
    return {alias for alias in aliases if alias and alias != "unknown"}


def check_research_readiness(
    *,
    runs_dir: str | Path = "artifacts/runs",
    external_eval_config: str | Path = "configs/benchmark/external_eval_vllm_template.yaml",
    reset_overlay: str | Path = "configs/benchmark/local_vllm_reset.yaml",
    model_family_config: str | Path = "configs/benchmark/model_family_vllm_template.yaml",
    capacity_frontier_path: str | Path = "data/results/capacity_frontier.yaml",
) -> list[ReadinessItem]:
    items: list[ReadinessItem] = []

    try:
        result = validate_workload_config(
            external_eval_config,
            min_records=500,
            required_tasks={"qa", "json", "summary", "code"},
            required_layouts={"document_before_instruction", "instruction_before_document"},
            balanced_tasks=True,
            min_shared_prefix_groups=4,
            require_tenant_fields=True,
        )
        items.append(
            _ok(
                "external_eval_preflight",
                f"{result['request_count']} records, tasks={result['task_counts']}",
            )
        )
    except Exception as exc:
        items.append(_missing("external_eval_preflight", str(exc)))

    try:
        external_config = load_config(external_eval_config)
        dataset_path = external_config.workload.dataset_path
        if _is_generated_external_fixture(dataset_path):
            items.append(
                _needs_evidence(
                    "independent_external_eval",
                    "external eval config still points at the generated public-domain-style "
                    "fixture; replace workload.dataset_path with independently curated JSONL",
                )
            )
        else:
            result = validate_workload_config(
                external_eval_config,
                min_records=500,
                required_tasks={"qa", "json", "summary", "code"},
                required_layouts={"document_before_instruction", "instruction_before_document"},
                balanced_tasks=True,
                min_shared_prefix_groups=4,
                require_tenant_fields=True,
                require_source_fields=True,
            )
            items.append(
                _ok(
                    "independent_external_eval",
                    f"{result['request_count']} source-provenance records",
                )
            )
    except Exception as exc:
        items.append(_missing("independent_external_eval", str(exc)))

    try:
        reset_config = load_config(reset_overlay)
        if reset_config.backend.server_reset_command and reset_config.backend.server_warmup_url:
            items.append(
                _ok(
                    "per_condition_vllm_cache_isolation",
                    "reset command and warmup URL are configured",
                )
            )
        else:
            items.append(
                _missing(
                    "per_condition_vllm_cache_isolation",
                    "reset overlay must set backend.server_reset_command and server_warmup_url",
                )
            )
    except Exception as exc:
        items.append(_missing("per_condition_vllm_cache_isolation", str(exc)))

    try:
        model_config = load_config(model_family_config)
        model_specs: list[dict[str, Any]] = list(model_config.matrix.get("models", []))
        configured = len(model_specs)
        complete_specs = _complete_model_specs(model_specs)
        observed_aliases = _observed_model_family_aliases(runs_dir)
        if len(observed_aliases) >= 2:
            items.append(
                _ok(
                    "multi_model_comparison",
                    f"{len(observed_aliases)} served model families observed",
                )
            )
        elif configured >= 2 and len(complete_specs) >= 2:
            items.append(
                _needs_evidence(
                    "multi_model_comparison",
                    "multi-family sweep config is complete; run at least two served "
                    "model families before claiming cross-family evidence",
                )
            )
        else:
            items.append(
                _needs_evidence(
                    "multi_model_comparison",
                    "sweep path is implemented; add and run a second trained adapter family "
                    "before claiming cross-family evidence",
                )
            )
    except Exception as exc:
        items.append(_missing("multi_model_comparison", str(exc)))

    try:
        capacity = load_capacity_frontier(capacity_frontier_path)
        has_failure = capacity["status"].eq("fails").any()
        has_success = capacity["status"].eq("starts").any()
        has_larger_gpu_success = (
            capacity["status"].eq("starts") & capacity["gpu_memory_gb"].ge(80)
        ).any()
        if has_failure and has_success and has_larger_gpu_success:
            items.append(
                _ok(
                    "capacity_frontier_evidence",
                    f"{len(capacity)} records with startup failures and larger-GPU success",
                )
            )
        else:
            items.append(
                _needs_evidence(
                    "capacity_frontier_evidence",
                    "record at least one startup failure, one startup success, and one "
                    "larger-GPU success",
                )
            )
    except Exception as exc:
        items.append(_missing("capacity_frontier_evidence", str(exc)))

    adapter_metrics = build_adapter_cache_metrics(runs_dir)
    if adapter_metrics.empty:
        items.append(
            _needs_evidence(
                "adapter_aware_serving_metrics",
                "postprocessor is implemented but no request artifacts were found",
            )
        )
    else:
        scopes = sorted(set(adapter_metrics["server_cache_metric_scope"]))
        items.append(
            _ok(
                "adapter_aware_serving_metrics",
                f"{len(adapter_metrics)} adapter/run rows; server metric scopes={scopes}",
            )
        )

    figure_paths = [
        Path("docs/figures/whitepaper_specialization_cache_tradeoff.png"),
        Path("docs/figures/quality_vs_p95_ttft.png"),
        Path("docs/figures/cache_hit_rate_by_policy_model.png"),
    ]
    report_paths = [
        Path("docs/claim_ladder.md"),
        Path("docs/real_eval_results.md"),
        Path("docs/benchmark_quality_plan.md"),
        Path("docs/research_plan.md"),
        Path("README.md"),
    ]
    missing_paths = [str(path) for path in [*figure_paths, *report_paths] if not path.exists()]
    if missing_paths:
        items.append(_missing("public_result_refresh", f"missing {missing_paths}"))
    else:
        items.append(
            _ok(
                "public_result_refresh",
                "README, claim ladder, benchmark-quality plan, real-eval notes, "
                "research plan, and public figures are present",
            )
        )

    return items


def format_markdown(items: list[ReadinessItem]) -> str:
    lines = ["| item | status | detail |", "| --- | --- | --- |"]
    for item in items:
        lines.append(f"| {item.name} | {item.status} | {item.detail} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()
    items = check_research_readiness(runs_dir=args.runs_dir)
    if args.format == "json":
        print(json.dumps([asdict(item) for item in items], indent=2))
    else:
        print(format_markdown(items))


if __name__ == "__main__":
    main()
