from __future__ import annotations

import argparse
from pathlib import Path

from specialization_cache_frontier.analysis.pareto import workload_pareto_frontiers
from specialization_cache_frontier.analysis.plots import generate_plots
from specialization_cache_frontier.analysis.slo import slo_sweep
from specialization_cache_frontier.bench.aggregate import (
    cache_model_means,
    layout_ablation_means,
    load_request_rows,
    load_summaries,
    repeated_seed_summary,
    router_means,
    workload_leaders,
    write_analysis_tables,
)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows available."]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return [header, divider, *body]


def generate_report(
    runs_dir: str | Path = "artifacts/runs",
    report_path: str | Path = "reports/specialization-cache-frontier.md",
    tables_dir: str | Path = "reports/tables",
) -> Path:
    df = load_summaries(runs_dir)
    request_df = load_request_rows(runs_dir)
    figures = generate_plots(df, request_df=request_df)
    table_paths = write_analysis_tables(df, request_df, tables_dir)
    leaders = workload_leaders(df)
    cache_means = cache_model_means(df)
    routers = router_means(df)
    repeated = repeated_seed_summary(df)
    layouts = layout_ablation_means(request_df)
    pareto = workload_pareto_frontiers(df)
    slo = slo_sweep(request_df)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        results = "No benchmark runs were found yet."
    else:
        best = df.sort_values("quality_adjusted_goodput", ascending=False).iloc[0]
        layout_note = ""
        if not request_df.empty and "workload" in request_df:
            layout_rows = request_df[request_df["workload"].eq("prompt_layout_ablation")]
        else:
            layout_rows = request_df
        if not layout_rows.empty:
            layout_means = layout_rows.groupby("prompt_layout")["ttft_ms"].mean().sort_index()
            layout_note = " ".join(
                f"`{layout}` mean TTFT is {ttft:.1f} ms." for layout, ttft in layout_means.items()
            )
        results = (
            f"Best quality-adjusted goodput in the current artifact set is "
            f"`{best['router_policy']}` with `{best['cache_model']}` on `{best['workload']}`. "
            f"Mean quality is {best['mean_quality']:.3f}, "
            f"p95 TTFT is {best['p95_ttft_ms']:.1f} ms, "
            f"and fragmentation index is {best['fragmentation_index']:.2f}. "
            f"{layout_note}".strip()
        )
    figure_lines = "\n".join(f"- `{path}`" for path in figures)
    table_lines = "\n".join(f"- `{path}`" for path in table_paths.values())
    leader_lines = _markdown_table(
        leaders.head(8).to_dict("records") if not leaders.empty else [],
        [
            "workload",
            "router_policy",
            "cache_model",
            "quality_adjusted_goodput",
            "mean_quality",
            "p95_ttft_ms",
        ],
    )
    cache_lines = _markdown_table(
        cache_means.head(8).to_dict("records") if not cache_means.empty else [],
        [
            "cache_model",
            "adapter_strategy",
            "quality_adjusted_goodput",
            "p95_ttft_ms",
            "cache_hit_rate",
            "fragmentation_index",
            "eviction_count",
        ],
    )
    router_lines = _markdown_table(
        routers.head(8).to_dict("records") if not routers.empty else [],
        ["router_policy", "quality_adjusted_goodput", "mean_quality", "p95_ttft_ms"],
    )
    repeated_lines = _markdown_table(
        repeated[repeated["run_count"] > 1].head(12).to_dict("records")
        if not repeated.empty
        else [],
        [
            "workload",
            "router_policy",
            "cache_model",
            "run_count",
            "quality_adjusted_goodput_mean",
            "quality_adjusted_goodput_std",
            "p95_ttft_ms_mean",
        ],
    )
    layout_lines = _markdown_table(
        layouts.head(12).to_dict("records") if not layouts.empty else [],
        ["prompt_layout", "cache_model", "ttft_ms", "quality", "cached_prompt_tokens"],
    )
    pareto_lines = _markdown_table(
        pareto.head(12).to_dict("records") if not pareto.empty else [],
        [
            "pareto_workload",
            "router_policy",
            "cache_model",
            "mean_quality",
            "p95_ttft_ms",
            "quality_adjusted_goodput",
        ],
    )
    slo_leaders = (
        slo.sort_values("quality_adjusted_goodput", ascending=False)
        .groupby("ttft_slo_ms", as_index=False)
        .head(1)
        if not slo.empty
        else slo
    )
    slo_lines = _markdown_table(
        slo_leaders.head(12).to_dict("records") if not slo_leaders.empty else [],
        [
            "ttft_slo_ms",
            "workload",
            "router_policy",
            "cache_model",
            "quality_adjusted_goodput",
            "requests_under_slo",
        ],
    )
    lines = [
        "# When is specialization worth its cache footprint?",
        "",
        "## Abstract",
        "",
        "Specialist adapters can improve task quality, but every adapter routing decision",
        "also changes the cache namespace seen by a causal transformer serving stack.",
        "This benchmark studies that tradeoff under shared-prefix workloads.",
        "",
        "## Core question",
        "",
        "When is model or adaptor specialization worth its KV-cache footprint?",
        "",
        "## Why this matters",
        "",
        "Naive semantic routing sends each task to its best specialist. Under repeated",
        "long prefixes, that can fragment prefix-cache reuse across adapters and increase",
        "TTFT even when the semantic choice is locally correct.",
        "",
        "## Experimental setup",
        "",
        "The first implementation uses a deterministic mock backend, whitespace",
        "tokenization, block prefix caching, and synthetic quality matrices.",
        "Real vLLM integration is intentionally optional.",
        "",
        "## Workloads",
        "",
        "The benchmark includes shared document QA, mixed tasks over the same document,",
        "multi-turn agent sessions, a low-overlap negative control, and prompt layout",
        "ablations.",
        "",
        "## Router policies",
        "",
        "Policies include random, semantic, sticky session, cache-aware, and oracle routing.",
        "",
        "## Cache models",
        "",
        "Models include standard LoRA-style adapter namespaces, optimistic base sharing,",
        "activated-LoRA-style late specialization, and copy-on-write deltas.",
        "",
        "## Results",
        "",
        results,
        "",
        "Generated figures:",
        "",
        figure_lines,
        "",
        "Generated tables:",
        "",
        table_lines,
        "",
        "### Workload leaders",
        "",
        *leader_lines,
        "",
        "### Cache-model means",
        "",
        *cache_lines,
        "",
        "### Router means",
        "",
        *router_lines,
        "",
        "### Repeated-seed summary",
        "",
        *repeated_lines,
        "",
        "### Prompt-layout ablation",
        "",
        *layout_lines,
        "",
        "### Pareto frontier",
        "",
        *pareto_lines,
        "",
        "### SLO sweep leaders",
        "",
        *slo_lines,
        "",
        "## Takeaways",
        "",
        "Specialization is most attractive when quality gains exceed the prefill and",
        "memory cost of lost prefix reuse. Cache-aware and late-specialization strategies",
        "recover locality without collapsing every task into one multitask adapter.",
        "",
        "## Limitations",
        "",
        "The default backend is a simulator. Tokenization, queueing, and quality are",
        "approximate. Real serving behavior should be validated with vLLM or another",
        "production server before making capacity decisions.",
        "",
        "## Physical AI analogue",
        "",
        "The same structure appears in VLA and robotics serving: repeated scene tokens",
        "map to a world-state cache, skill adapters map to embodiment or task",
        "specialization, and goodput maps to success-rate-adjusted control Hz.",
        "",
    ]
    text = "\n".join(lines)
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--report-path", default="reports/specialization-cache-frontier.md")
    parser.add_argument("--tables-dir", default="reports/tables")
    args = parser.parse_args()
    print(generate_report(args.runs_dir, args.report_path, args.tables_dir))


if __name__ == "__main__":
    main()
