from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adapter_cache_bench.analysis.adapter_cache_metrics import build_adapter_cache_metrics
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


def check_research_readiness(
    *,
    runs_dir: str | Path = "artifacts/runs",
    external_eval_config: str | Path = "configs/benchmark/external_eval_vllm_template.yaml",
    reset_overlay: str | Path = "configs/benchmark/local_vllm_reset.yaml",
    model_family_config: str | Path = "configs/benchmark/model_family_vllm_template.yaml",
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
        complete_specs = [
            spec
            for spec in model_specs
            if isinstance(spec, dict)
            and {"qa", "json", "summary", "code", "multitask"}
            <= set((spec.get("adapter_model_names") or {}).keys())
        ]
        if configured >= 2 and len(complete_specs) >= 2:
            items.append(_ok("multi_model_comparison", f"{configured} model families configured"))
        else:
            items.append(
                _needs_evidence(
                    "multi_model_comparison",
                    "sweep path is implemented; add a second trained adapter family before "
                    "claiming cross-family evidence",
                )
            )
    except Exception as exc:
        items.append(_missing("multi_model_comparison", str(exc)))

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
        Path("docs/real_eval_results.md"),
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
                "README, real-eval notes, research plan, and public figures are present",
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
