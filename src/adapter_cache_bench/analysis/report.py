from __future__ import annotations

import argparse
from pathlib import Path

from adapter_cache_bench.analysis.pareto import workload_pareto_frontiers
from adapter_cache_bench.analysis.plots import generate_plots
from adapter_cache_bench.analysis.regime_science import write_regime_policy_failure_map
from adapter_cache_bench.analysis.slo import slo_sweep
from adapter_cache_bench.bench.aggregate import (
    cache_model_means,
    layout_ablation_means,
    load_request_rows,
    load_summaries,
    repeated_seed_summary,
    router_means,
    workload_leaders,
    write_analysis_tables,
)


def _large_model_overlap_claim(df) -> list[str]:
    if df.empty or "run_id" not in df:
        return ["No reset-isolated 7B overlap sweep is present in the artifact set."]
    sub = df[df["run_id"].str.contains("large-model-overlap-confidence-vllm-streaming", na=False)]
    required = {
        "sweep_overlap_fraction",
        "run_id",
        "request_count",
        "p95_ttft_ms",
        "slo_attainment_rate",
        "quality_adjusted_goodput",
        "server_prefix_cache_hit_rate",
    }
    if sub.empty or not required <= set(sub.columns):
        return ["No reset-isolated 7B overlap sweep is present in the artifact set."]
    sub = sub.sort_values("run_id").drop_duplicates(
        ["sweep_overlap_fraction", "sweep_seed"],
        keep="last",
    )
    grouped = (
        sub.groupby("sweep_overlap_fraction", as_index=False)
        .agg(
            runs=("run_id", "count"),
            requests=("request_count", "sum"),
            p95_ttft_ms=("p95_ttft_ms", "mean"),
            slo_attainment_rate=("slo_attainment_rate", "mean"),
            quality_adjusted_goodput=("quality_adjusted_goodput", "mean"),
            server_prefix_cache_hit_rate=("server_prefix_cache_hit_rate", "mean"),
        )
        .sort_values("sweep_overlap_fraction")
    )
    if len(grouped) < 2:
        return ["The 7B overlap sweep is present but has fewer than two overlap levels."]
    low = grouped.iloc[0]
    high = grouped.iloc[-1]
    p95_delta = low["p95_ttft_ms"] - high["p95_ttft_ms"]
    p95_pct = p95_delta / low["p95_ttft_ms"] * 100
    hit_delta = (high["server_prefix_cache_hit_rate"] - low["server_prefix_cache_hit_rate"]) * 100
    slo_delta = (high["slo_attainment_rate"] - low["slo_attainment_rate"]) * 100
    qag_delta = (
        (high["quality_adjusted_goodput"] - low["quality_adjusted_goodput"])
        / low["quality_adjusted_goodput"]
        * 100
    )
    return [
        (
            f"- Reset-isolated 7B cache locality: {int(grouped['requests'].sum())} "
            f"requests across {int(grouped['runs'].sum())} runs. Moving from "
            f"{low['sweep_overlap_fraction']:.0%} to {high['sweep_overlap_fraction']:.0%} "
            f"shared-prefix overlap raised server prefix-cache hit rate by "
            f"{hit_delta:.1f} percentage points, reduced mean p95 TTFT by "
            f"{p95_delta:.1f} ms ({p95_pct:.1f}%), lifted SLO attainment by "
            f"{slo_delta:.1f} percentage points, and raised QAG by {qag_delta:.1f}%."
        )
    ]


def _has_workload_prefix(df, prefix: str) -> bool:
    if df.empty or "workload" not in df:
        return False
    return df["workload"].astype(str).str.startswith(prefix, na=False).any()


def _filter_backend_rows(df, backend_kind: str):
    if df.empty or "backend_kind" not in df:
        return df.iloc[0:0]
    return df[df["backend_kind"].astype(str).eq(backend_kind)]


def _filter_workload_prefix_rows(df, prefix: str):
    if df.empty or "workload" not in df:
        return df.iloc[0:0]
    return df[df["workload"].astype(str).str.startswith(prefix, na=False)]


def _filter_backend_workload_prefix_rows(df, backend_kind: str, prefix: str):
    return _filter_workload_prefix_rows(_filter_backend_rows(df, backend_kind), prefix)


def _has_cache_conditions(df, expected: set[str]) -> bool:
    if df.empty or "cache_condition" not in df:
        return False
    observed = set(df["cache_condition"].dropna().astype(str))
    return expected <= observed


def _has_positive_server_cache_counters(df) -> bool:
    if df.empty:
        return False
    counter_columns = [
        "server_prefix_cache_queries",
        "server_prefix_cache_hits",
        "server_prompt_tokens_cached",
    ]
    for column in counter_columns:
        if column in df and (df[column].fillna(0).astype(float) > 0).any():
            return True
    return False


def _claim_boundary_lines(df) -> list[str]:
    regime_mock_rows = _filter_backend_workload_prefix_rows(df, "mock", "regime_")
    regime_vllm_rows = _filter_backend_workload_prefix_rows(df, "vllm", "regime_")
    has_regime_mock = not regime_mock_rows.empty
    has_regime_vllm = not regime_vllm_rows.empty
    has_cache_controls = _has_cache_conditions(
        regime_mock_rows,
        {"warm", "cold", "prefix_disabled"},
    )
    has_server_counters = _has_positive_server_cache_counters(regime_vllm_rows)
    rows = [
        {
            "Claim area": "Simulator regime map",
            "Status": "supported in this report" if has_regime_mock else "not present here",
            "Evidence in this report": (
                "mock `regime_*` runs with structure metrics and policy regret"
                if has_regime_mock
                else "run `configs/benchmark/regime_v0_mock.yaml`"
            ),
            "Required before widening": "reset-isolated real-server bridge",
        },
        {
            "Claim area": "Cache-control mechanisms",
            "Status": "simulator-backed" if has_cache_controls else "incomplete controls",
            "Evidence in this report": (
                "`warm`, `cold`, and `prefix_disabled` rows"
                if has_cache_controls
                else "add warm, cold, and prefix-disabled cache conditions"
            ),
            "Required before widening": "server reset settings and cache-counter provenance",
        },
        {
            "Claim area": "Real-server regime bridge",
            "Status": "candidate evidence present" if has_regime_vllm else "not supported here",
            "Evidence in this report": (
                "vLLM rows over `regime_*` workloads"
                if has_regime_vllm
                else "no reset-isolated vLLM regime sweep in this artifact set"
            ),
            "Required before widening": "repeat claim-critical regimes with comparable conditions",
        },
        {
            "Claim area": "Prefix-cache causality",
            "Status": "counter-backed" if has_server_counters else "not established here",
            "Evidence in this report": (
                "positive server-side prefix/cache counters"
                if has_server_counters
                else "no positive server-side prefix/cache counters"
            ),
            "Required before widening": "capture counters or downgrade to client-observed behavior",
        },
        {
            "Claim area": "Automated recommendations",
            "Status": "deferred",
            "Evidence in this report": "policy comparisons are explanatory, not prescriptive",
            "Required before widening": "G8 bridge plus uncertainty and user-path readiness",
        },
    ]
    return _markdown_table(
        rows,
        [
            "Claim area",
            "Status",
            "Evidence in this report",
            "Required before widening",
        ],
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


def _interpretation_lines(cache_means, layouts, repeated, df) -> list[str]:
    lines = []
    if not cache_means.empty:
        best_cache = cache_means.sort_values(
            "quality_adjusted_goodput",
            ascending=False,
        ).iloc[0]
        lines.append(
            f"- Best aggregate cache strategy: `{best_cache['cache_model']}` "
            f"with mean quality-adjusted goodput {best_cache['quality_adjusted_goodput']:.3f}."
        )
        if "quality_adjusted_goodput_per_memory_token" in cache_means:
            efficient = cache_means.sort_values(
                "quality_adjusted_goodput_per_memory_token",
                ascending=False,
            ).iloc[0]
            lines.append(
                f"- Best cache-footprint efficiency: `{efficient['cache_model']}` "
                "with mean quality-adjusted goodput per memory token "
                f"{efficient['quality_adjusted_goodput_per_memory_token']:.6f}."
            )
    if not layouts.empty:
        layout_means = layouts.groupby("prompt_layout")["ttft_ms"].mean()
        if {
            "document_before_instruction",
            "instruction_before_document",
        } <= set(layout_means.index):
            document_first = layout_means["document_before_instruction"]
            instruction_first = layout_means["instruction_before_document"]
            delta = instruction_first - document_first
            lines.append(
                "- Prompt layout matters: `document_before_instruction` is "
                f"{delta:.1f} ms lower mean TTFT than `instruction_before_document` "
                "in the current artifact set."
            )
    if not df.empty and "eviction_count" in df:
        eviction_runs = df[df["eviction_count"] > 0]
        if eviction_runs.empty:
            lines.append(
                "- No committed baseline run evicted cache blocks; use the memory-pressure "
                "matrix to exercise finite-cache behavior."
            )
        else:
            worst = eviction_runs.sort_values("eviction_count", ascending=False).iloc[0]
            lines.append(
                f"- Highest eviction pressure: `{worst['cache_model']}` on "
                f"`{worst['workload']}` with {int(worst['eviction_count'])} evictions."
            )
    if not repeated.empty and (repeated["run_count"] > 1).any():
        best_repeated = repeated[repeated["run_count"] > 1].iloc[0]
        lines.append(
            f"- Repeated-seed leader: `{best_repeated['router_policy']}` with "
            f"`{best_repeated['cache_model']}` on `{best_repeated['workload']}`."
        )
    else:
        lines.append(
            "- Repeated-seed support is available; run `configs/benchmark/repeated.yaml` "
            "to populate variance rows."
        )
    return lines or ["- No interpretation available until benchmark summaries are present."]


def _evidence_lines(df) -> list[str]:
    if df.empty:
        return ["No benchmark evidence is present yet."]
    if "backend_kind" not in df:
        return ["Backend provenance is unavailable for these runs."]
    rows = []
    grouped = (
        df.fillna({"backend_kind": "unknown", "backend_model": "unknown"})
        .groupby(["backend_kind", "backend_model"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            requests=("request_count", "sum"),
            mean_quality=("mean_quality", "mean"),
            p95_ttft_ms=("p95_ttft_ms", "mean"),
        )
        .sort_values(["backend_kind", "backend_model"])
    )
    for row in grouped.to_dict("records"):
        rows.append(
            f"- `{row['backend_kind']}` / `{row['backend_model']}`: "
            f"{int(row['runs'])} runs, {int(row['requests'])} requests, "
            f"mean quality {row['mean_quality']:.3f}, "
            f"mean p95 TTFT {row['p95_ttft_ms']:.1f} ms."
        )
    return rows


def generate_report(
    runs_dir: str | Path = "artifacts/runs",
    report_path: str | Path = "reports/adapter-cache-tradeoffs.md",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
) -> Path:
    df = load_summaries(runs_dir)
    request_df = load_request_rows(runs_dir)
    figures = generate_plots(df, output_dir=figures_dir, request_df=request_df)
    table_paths = write_analysis_tables(df, request_df, tables_dir, runs_dir=runs_dir)
    regime_figure = write_regime_policy_failure_map(
        runs_dir,
        Path(figures_dir) / "regime_policy_failure_map.png",
    )
    if regime_figure is not None:
        figures.append(regime_figure)
    leaders = workload_leaders(df)
    cache_means = cache_model_means(df)
    routers = router_means(df)
    repeated = repeated_seed_summary(df)
    layouts = layout_ablation_means(request_df)
    pareto = workload_pareto_frontiers(df)
    slo = slo_sweep(request_df)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    claim_ladder_link = (
        "claim_ladder.md" if report.parent.name == "docs" else "../docs/claim_ladder.md"
    )
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
            "cache_condition",
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
            "cache_condition",
            "adapter_strategy",
            "quality_adjusted_goodput",
            "quality_adjusted_goodput_per_memory_token",
            "p95_ttft_ms",
            "cache_hit_rate",
            "fragmentation_index",
            "eviction_count",
        ],
    )
    router_lines = _markdown_table(
        routers.head(8).to_dict("records") if not routers.empty else [],
        [
            "router_policy",
            "cache_condition",
            "quality_adjusted_goodput",
            "quality_adjusted_goodput_per_memory_token",
            "mean_quality",
            "p95_ttft_ms",
        ],
    )
    repeated_lines = _markdown_table(
        repeated[repeated["run_count"] > 1].head(12).to_dict("records")
        if not repeated.empty
        else [],
        [
            "workload",
            "cache_condition",
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
    interpretation = _interpretation_lines(cache_means, layouts, repeated, df)
    evidence = _evidence_lines(df)
    overlap_claim = _large_model_overlap_claim(df)
    claim_boundary = _claim_boundary_lines(df)
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
        "The deterministic mock backend is a systems simulator: it isolates routing,",
        "prefix-cache locality, finite KV budget, eviction, prompt layout, and SLO",
        "behavior from GPU and serving noise. Real model-server runs use the same",
        "JSONL workloads with `backend.kind: vllm` or another OpenAI-compatible",
        "backend.",
        "",
        "### Evidence classes",
        "",
        *evidence,
        "",
        "### Claim ladder",
        "",
        "The maintained public claim boundary lives in",
        f"[docs/claim_ladder.md]({claim_ladder_link}). Current supported claims",
        "must cite model/server, request count, run count, and metric scope.",
        "",
        *overlap_claim,
        "",
        "### Claim boundary",
        "",
        "The report separates simulator-backed findings from real-serving claims.",
        "Treat missing gates as scope limits, not as negative results.",
        "",
        *claim_boundary,
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
        "### Decision rule",
        "",
        "Treat specialization as worthwhile only when it improves quality-adjusted",
        "goodput under the TTFT SLO after accounting for cache memory. In this",
        "repo that means comparing both `quality_adjusted_goodput` and",
        "`quality_adjusted_goodput_per_memory_token`, while checking that the",
        "fragmentation index and SLO attainment do not regress beyond the serving",
        "budget for the workload.",
        "",
        "### Interpretation",
        "",
        *interpretation,
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
    parser.add_argument("--report-path", default="reports/adapter-cache-tradeoffs.md")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()
    print(generate_report(args.runs_dir, args.report_path, args.tables_dir, args.figures_dir))


if __name__ == "__main__":
    main()
