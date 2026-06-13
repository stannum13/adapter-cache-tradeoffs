from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_summaries(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(runs_dir).glob("*/summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return pd.DataFrame(rows)


def load_request_rows(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(runs_dir).glob("*/requests.jsonl"):
        summary_path = path.parent / "summary.json"
        metadata = {}
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            metadata = {
                "run_id": summary["run_id"],
                "router_policy": summary["router_policy"],
                "cache_model": summary["cache_model"],
                "workload": summary["workload"],
            }
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                request = record["request"]
                response = record["response"]
                metrics = response["metrics"]
                quality = response["quality"]
                routing = record["routing"]
                rows.append(
                    {
                        **metadata,
                        "request_id": request["request_id"],
                        "prompt_layout": request["prompt_layout"],
                        "task_type": request["task_type"],
                        "adapter_id": routing["adapter_id"],
                        "ttft_ms": metrics["ttft_ms"],
                        "e2e_ms": metrics["e2e_ms"],
                        "cached_prompt_tokens": metrics["cached_prompt_tokens"],
                        "prompt_tokens": metrics["prompt_tokens"],
                        "quality": quality["score"],
                    }
                )
    return pd.DataFrame(rows)
