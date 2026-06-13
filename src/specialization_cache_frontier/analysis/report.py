from __future__ import annotations

import argparse
from pathlib import Path

from specialization_cache_frontier.analysis.plots import generate_plots
from specialization_cache_frontier.bench.aggregate import load_summaries


def generate_report(
    runs_dir: str | Path = "artifacts/runs",
    report_path: str | Path = "reports/specialization-cache-frontier.md",
) -> Path:
    df = load_summaries(runs_dir)
    figures = generate_plots(df)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        results = "No benchmark runs were found yet."
    else:
        best = df.sort_values("quality_adjusted_goodput", ascending=False).iloc[0]
        results = (
            f"Best quality-adjusted goodput in the current artifact set is "
            f"`{best['router_policy']}` with `{best['cache_model']}` on `{best['workload']}`. "
            f"Mean quality is {best['mean_quality']:.3f}, "
            f"p95 TTFT is {best['p95_ttft_ms']:.1f} ms, "
            f"and fragmentation index is {best['fragmentation_index']:.2f}."
        )
    figure_lines = "\n".join(f"- `{path}`" for path in figures)
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
    args = parser.parse_args()
    print(generate_report(args.runs_dir, args.report_path))


if __name__ == "__main__":
    main()
