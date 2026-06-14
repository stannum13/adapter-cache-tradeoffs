from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from adapter_cache_bench.config import CacheConfig, WorkloadConfig
from adapter_cache_bench.routing.scoring import expected_adapter_for_task
from adapter_cache_bench.types import RequestRecord
from adapter_cache_bench.workloads.documents import make_document
from adapter_cache_bench.workloads.synthetic_ground_truth import ground_truth_for
from adapter_cache_bench.workloads.templates import prompt_for


def generate_workload(
    config: WorkloadConfig, cache_config: CacheConfig | None = None
) -> list[RequestRecord]:
    if config.name == "shared_doc_qa":
        return list(_shared_doc_qa(config, cache_config))
    if config.name == "mixed_tasks_same_doc":
        return list(_mixed_tasks_same_doc(config, cache_config))
    if config.name == "agent_session":
        return list(_agent_session(config, cache_config))
    if config.name == "low_overlap_control":
        return list(_low_overlap_control(config, cache_config))
    if config.name == "prompt_layout_ablation":
        return list(_prompt_layout_ablation(config, cache_config))
    if config.name == "jsonl_eval":
        return list(_jsonl_eval(config, cache_config))
    raise ValueError(f"Unknown workload: {config.name}")


def _base_fields(config: WorkloadConfig, i: int, task_type: str, document_id: int):
    return {
        "request_id": f"{config.name}-{i:05d}",
        "session_id": f"session-{i % max(1, config.sessions)}",
        "tenant_id": f"tenant-{i % max(1, config.tenants)}",
        "trust_group_id": f"trust-{(i // 2) % max(1, config.tenants)}",
        "task_type": task_type,
        "shared_prefix_id": f"doc-{document_id}",
        "expected_adapter": expected_adapter_for_task(task_type),
        "ground_truth": ground_truth_for(task_type, document_id, i),
        "max_tokens": config.max_tokens,
        "requires_json": task_type == "json",
    }


def _shared_doc_qa(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    docs = [make_document(i, config.document_tokens) for i in range(config.shared_document_count)]
    for i in range(config.request_count):
        document_id = i % len(docs)
        question = f"What recorded fact answers question {i}?"
        prompt = prompt_for(
            "qa",
            docs[document_id],
            question,
            "document_before_instruction",
            "qa",
            True,
            cache_config,
        )
        yield RequestRecord(
            **_base_fields(config, i, "qa", document_id),
            prompt=prompt,
            prompt_layout="document_before_instruction",
        )


def _mixed_tasks_same_doc(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    tasks = ["qa", "json", "summary", "code"]
    document = make_document(0, config.document_tokens)
    for i in range(config.request_count):
        task = tasks[i % len(tasks)]
        question = f"Perform {task} operation number {i} over the shared document."
        adapter = expected_adapter_for_task(task)
        prompt = prompt_for(
            task, document, question, "document_before_instruction", adapter, True, cache_config
        )
        yield RequestRecord(
            **_base_fields(config, i, task, 0),
            prompt=prompt,
            prompt_layout="document_before_instruction",
        )


def _agent_session(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    doc = make_document(0, config.document_tokens // 2)
    histories = {
        f"session-{i}": [f"System trace: loaded tools for session {i}."]
        for i in range(config.sessions)
    }
    tasks = ["qa", "json", "summary"]
    for i in range(config.request_count):
        session_id = f"session-{i % max(1, config.sessions)}"
        task = tasks[i % len(tasks)]
        histories[session_id].append(f"Tool trace {i}: searched document and read stable facts.")
        history = " ".join(histories[session_id])
        prompt = prompt_for(
            task,
            f"{doc} Repeated history: {history}",
            f"Continue turn {i}.",
            "document_before_instruction",
            expected_adapter_for_task(task),
            True,
            cache_config,
        )
        fields = _base_fields(config, i, task, 0)
        fields["session_id"] = session_id
        yield RequestRecord(**fields, prompt=prompt, prompt_layout="document_before_instruction")


def _low_overlap_control(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    rng = random.Random(config.seed)
    tasks = ["qa", "json", "summary", "code"]
    for i in range(config.request_count):
        task = tasks[i % len(tasks)]
        random_doc = " ".join(
            f"rand_{i}_{rng.randrange(1_000_000)}" for _ in range(config.document_tokens)
        )
        prompt = prompt_for(
            task,
            random_doc,
            f"Unique request {i}",
            "instruction_before_document",
            expected_adapter_for_task(task),
            True,
            cache_config,
        )
        yield RequestRecord(
            **_base_fields(config, i, task, i),
            prompt=prompt,
            prompt_layout="instruction_before_document",
        )


def _prompt_layout_ablation(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    document = make_document(0, config.document_tokens)
    layouts = ["instruction_before_document", "document_before_instruction"]
    tasks = ["qa", "json"]
    for i in range(config.request_count):
        task = tasks[i % len(tasks)]
        layout = layouts[i % len(layouts)]
        adapter = expected_adapter_for_task(task)
        prompt = prompt_for(
            task,
            document,
            f"Equivalent layout request {i}",
            layout,
            adapter,
            True,
            cache_config,
        )
        yield RequestRecord(**_base_fields(config, i, task, 0), prompt=prompt, prompt_layout=layout)


def _read_jsonl_or_yaml(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or []
        if not isinstance(data, list):
            raise ValueError(f"Expected list in workload dataset: {path}")
        return data
    rows = []
    for line in text.splitlines():
        if line.strip():
            import json

            rows.append(json.loads(line))
    return rows


def _jsonl_eval(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    if not config.dataset_path:
        raise ValueError("jsonl_eval workload requires workload.dataset_path")
    rows = _read_jsonl_or_yaml(Path(config.dataset_path))
    for i, row in enumerate(rows[: config.request_count]):
        task = row["task_type"]
        document = row["document"]
        adapter = row.get("expected_adapter") or expected_adapter_for_task(task)
        layout = row.get("prompt_layout", "document_before_instruction")
        prompt = row.get("prompt") or prompt_for(
            task,
            document,
            row.get("question", f"Evaluate record {i}."),
            layout,
            adapter,
            True,
            cache_config,
        )
        yield RequestRecord(
            request_id=row.get("request_id", f"jsonl-eval-{i:05d}"),
            session_id=row.get("session_id", f"eval-session-{i % max(1, config.sessions)}"),
            tenant_id=row.get("tenant_id", "eval-tenant"),
            trust_group_id=row.get("trust_group_id", "eval-trust"),
            task_type=task,
            prompt=prompt,
            shared_prefix_id=row.get("shared_prefix_id", row.get("document_id", "eval-doc")),
            expected_adapter=adapter,
            ground_truth=row.get("ground_truth"),
            max_tokens=int(row.get("max_tokens", config.max_tokens)),
            prompt_layout=layout,
            requires_json=bool(row.get("requires_json", task == "json")),
        )
