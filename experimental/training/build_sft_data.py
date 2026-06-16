from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from adapter_cache_bench.config import load_config
from adapter_cache_bench.types import RequestRecord
from adapter_cache_bench.workloads.generator import generate_workload


def completion_for(record: RequestRecord) -> str:
    if record.task_type == "json":
        return json.dumps(record.ground_truth, sort_keys=True, separators=(",", ":"))
    if record.task_type == "code" and isinstance(record.ground_truth, dict):
        tests = record.ground_truth.get("tests", [])
        return "\n".join(str(test) for test in tests)
    return str(record.ground_truth or "")


def sft_row(record: RequestRecord) -> dict[str, Any]:
    return {
        "request_id": record.request_id,
        "task_type": record.task_type,
        "adapter_id": record.expected_adapter,
        "prompt": record.prompt,
        "completion": completion_for(record),
    }


def request_row(record: RequestRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_sft_split(
    config_paths: list[str],
    output_dir: str | Path,
    eval_fraction: float = 0.2,
    seed: int = 17,
) -> dict[str, Path]:
    config = load_config(config_paths)
    records = generate_workload(config.workload, config.cache)
    rng = random.Random(seed)
    rng.shuffle(records)
    eval_count = max(1, int(round(len(records) * eval_fraction))) if len(records) > 1 else 0
    eval_records = records[:eval_count]
    train_records = records[eval_count:]
    train_rows = [sft_row(record) for record in train_records]
    eval_rows = [sft_row(record) for record in eval_records]
    output = Path(output_dir)
    paths: dict[str, Path] = {}
    paths["train"] = output / "train.jsonl"
    paths["eval"] = output / "eval.jsonl"
    paths["train_requests"] = output / "train_requests.jsonl"
    paths["eval_requests"] = output / "eval_requests.jsonl"
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["eval"], eval_rows)
    write_jsonl(paths["train_requests"], [request_row(record) for record in train_records])
    write_jsonl(paths["eval_requests"], [request_row(record) for record in eval_records])
    all_rows = train_rows + eval_rows
    for task in sorted({row["task_type"] for row in all_rows}):
        task_train = [row for row in train_rows if row["task_type"] == task]
        task_eval = [row for row in eval_rows if row["task_type"] == task]
        paths[f"train_{task}"] = output / f"train_{task}.jsonl"
        paths[f"eval_{task}"] = output / f"eval_{task}.jsonl"
        write_jsonl(paths[f"train_{task}"], task_train)
        write_jsonl(paths[f"eval_{task}"], task_eval)
    metadata = {
        "config_paths": config_paths,
        "row_count": len(all_rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "tasks": sorted({row["task_type"] for row in all_rows}),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["metadata"] = output / "metadata.json"
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-config", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    paths = build_sft_split(
        args.workload_config,
        args.output_dir,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
