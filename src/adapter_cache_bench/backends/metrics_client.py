from __future__ import annotations

import math
import re

import httpx

METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)"
)


def parse_prometheus_samples(metrics_text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if not match:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        name = match.group("name")
        samples[name] = samples.get(name, 0.0) + value
    return samples


def prometheus_delta(before_text: str, after_text: str) -> dict[str, float]:
    before = parse_prometheus_samples(before_text)
    after = parse_prometheus_samples(after_text)
    names = set(before) | set(after)
    return {name: after.get(name, 0.0) - before.get(name, 0.0) for name in sorted(names)}


class MetricsClient:
    def __init__(self, metrics_url: str = "http://localhost:8000/metrics") -> None:
        self.metrics_url = metrics_url

    def scrape(self) -> str:
        return httpx.get(self.metrics_url, timeout=5.0).text

    def parse_scalar(self, metrics_text: str, metric_name: str) -> float | None:
        for line in metrics_text.splitlines():
            if line.startswith("#") or not line.startswith(metric_name):
                continue
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                return None
        return None

    def parse_samples(self, metrics_text: str) -> dict[str, float]:
        return parse_prometheus_samples(metrics_text)

    def delta(self, before_text: str, after_text: str) -> dict[str, float]:
        return prometheus_delta(before_text, after_text)
