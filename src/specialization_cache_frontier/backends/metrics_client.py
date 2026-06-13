from __future__ import annotations

import httpx


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
