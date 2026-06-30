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

REGIME_TASKS = ("qa", "json", "summary", "code")


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
    if config.name == "controlled_overlap":
        return list(_controlled_overlap(config, cache_config))
    if config.name == "prompt_layout_ablation":
        return list(_prompt_layout_ablation(config, cache_config))
    if config.name == "jsonl_eval":
        return list(_jsonl_eval(config, cache_config))
    if config.name == "regime_uniform":
        return list(_regime_uniform(config, cache_config))
    if config.name == "regime_zipfian":
        return list(_regime_zipfian(config, cache_config))
    if config.name == "regime_bursty_session":
        return list(_regime_bursty_session(config, cache_config))
    if config.name == "regime_phase_shift":
        return list(_regime_phase_shift(config, cache_config))
    if config.name == "regime_adversarial_churn":
        return list(_regime_adversarial_churn(config, cache_config))
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


def _controlled_overlap(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    rng = random.Random(config.seed)
    tasks = ["qa", "json", "summary", "code"]
    shared_count = round(config.document_tokens * max(0.0, min(1.0, config.shared_prefix_fraction)))
    unique_count = max(0, config.document_tokens - shared_count)
    shared_prefix = " ".join(f"shared_{token}" for token in range(shared_count))
    for i in range(config.request_count):
        task = tasks[i % len(tasks)]
        unique_suffix = " ".join(
            f"unique_{i}_{rng.randrange(1_000_000)}" for _ in range(unique_count)
        )
        document = f"{shared_prefix} {unique_suffix}".strip()
        prompt = prompt_for(
            task,
            document,
            f"Controlled overlap request {i}",
            "document_before_instruction",
            expected_adapter_for_task(task),
            True,
            cache_config,
        )
        yield RequestRecord(
            **_base_fields(config, i, task, i),
            prompt=prompt,
            prompt_layout="document_before_instruction",
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


def _balanced_task_sequence(count: int, rng: random.Random) -> list[str]:
    sequence = [REGIME_TASKS[i % len(REGIME_TASKS)] for i in range(count)]
    rng.shuffle(sequence)
    return sequence


def _weighted_task_sequence(count: int, weighted_tasks: list[str], rng: random.Random) -> list[str]:
    sequence: list[str] = []
    while len(sequence) < count:
        block = list(weighted_tasks)
        rng.shuffle(block)
        sequence.extend(block)
    return sequence[:count]


def _doc_picker(config: WorkloadConfig):
    documents: dict[int, str] = {}

    def pick(document_id: int) -> str:
        if document_id not in documents:
            documents[document_id] = make_document(document_id, config.document_tokens)
        return documents[document_id]

    return pick


def _regime_record(
    config: WorkloadConfig,
    cache_config: CacheConfig | None,
    i: int,
    task: str,
    document: str,
    document_id: int,
    session_id: str,
    question: str,
    layout: str = "document_before_instruction",
) -> RequestRecord:
    prompt = prompt_for(
        task,
        document,
        question,
        layout,
        expected_adapter_for_task(task),
        True,
        cache_config,
    )
    fields = _base_fields(config, i, task, document_id)
    fields["session_id"] = session_id
    return RequestRecord(**fields, prompt=prompt, prompt_layout=layout)


def _regime_uniform(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    rng = random.Random(config.seed)
    doc_count = max(1, config.shared_document_count)
    document_for = _doc_picker(config)
    task_sequence = _balanced_task_sequence(config.request_count, rng)
    document_sequence = [i % doc_count for i in range(config.request_count)]
    rng.shuffle(document_sequence)
    for i, (task, document_id) in enumerate(zip(task_sequence, document_sequence, strict=True)):
        session_id = f"session-{rng.randrange(max(1, config.sessions))}"
        question = f"Uniform regime request {i}: apply {task} analysis to document {document_id}."
        yield _regime_record(
            config,
            cache_config,
            i,
            task,
            document_for(document_id),
            document_id,
            session_id,
            question,
        )


def _regime_zipfian(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    rng = random.Random(config.seed)
    doc_count = max(1, config.shared_document_count)
    document_for = _doc_picker(config)
    task_sequence = _weighted_task_sequence(
        config.request_count,
        ["qa"] * 12 + ["json"] * 6 + ["summary"] * 3 + ["code"],
        rng,
    )
    document_weights = [max(1, doc_count - rank) for rank in range(doc_count)]
    document_ids = list(range(doc_count))
    for i, task in enumerate(task_sequence):
        document_id = rng.choices(document_ids, weights=document_weights, k=1)[0]
        hot_session_count = max(1, min(config.sessions, 3))
        session_id = f"session-{rng.randrange(hot_session_count)}"
        question = (
            f"Zipfian regime request {i}: use the recurring document {document_id} "
            f"for a {task} result."
        )
        yield _regime_record(
            config,
            cache_config,
            i,
            task,
            document_for(document_id),
            document_id,
            session_id,
            question,
        )


def _regime_bursty_session(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    rng = random.Random(config.seed)
    session_count = max(1, config.sessions)
    doc_count = max(1, config.shared_document_count)
    document_for = _doc_picker(config)
    session_order = list(range(session_count))
    rng.shuffle(session_order)
    i = 0
    burst_index = 0
    while i < config.request_count:
        session_number = session_order[burst_index % session_count]
        session_id = f"session-{session_number}"
        document_id = (session_number + burst_index) % doc_count
        burst_length = min(config.request_count - i, rng.randint(4, 8))
        task_offset = rng.randrange(len(REGIME_TASKS))
        for turn in range(burst_length):
            task = REGIME_TASKS[(task_offset + turn) % len(REGIME_TASKS)]
            question = (
                f"Bursty session {session_number} turn {turn}: continue work on "
                f"document {document_id} with {task}."
            )
            yield _regime_record(
                config,
                cache_config,
                i,
                task,
                document_for(document_id),
                document_id,
                session_id,
                question,
            )
            i += 1
        burst_index += 1


def _regime_phase_shift(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    rng = random.Random(config.seed)
    doc_count = max(4, config.shared_document_count)
    document_for = _doc_picker(config)
    midpoint = config.request_count // 2
    first_phase_docs = list(range(doc_count // 2))
    second_phase_docs = list(range(doc_count // 2, doc_count))
    first_phase_tasks = _weighted_task_sequence(
        midpoint,
        ["qa"] * 6 + ["json"] * 4 + ["summary"],
        rng,
    )
    second_phase_tasks = _weighted_task_sequence(
        config.request_count - midpoint,
        ["code"] * 6 + ["summary"] * 4 + ["json"],
        rng,
    )
    for i, task in enumerate(first_phase_tasks + second_phase_tasks):
        docs = first_phase_docs if i < midpoint else second_phase_docs
        document_id = docs[(i + rng.randrange(len(docs))) % len(docs)]
        session_id = f"session-{i % max(1, config.sessions)}"
        phase_name = "alpha" if i < midpoint else "beta"
        question = (
            f"Phase-shift regime {phase_name} request {i}: produce the {task} "
            f"output for document {document_id}."
        )
        yield _regime_record(
            config,
            cache_config,
            i,
            task,
            document_for(document_id),
            document_id,
            session_id,
            question,
        )


def _regime_adversarial_churn(
    config: WorkloadConfig, cache_config: CacheConfig | None
) -> Iterable[RequestRecord]:
    rng = random.Random(config.seed)
    task_order = list(REGIME_TASKS)
    rng.shuffle(task_order)
    doc_count = max(config.request_count, config.shared_document_count, 1)
    session_count = max(1, config.sessions)
    document_for = _doc_picker(config)
    document_offset = rng.randrange(doc_count)
    session_offset = rng.randrange(session_count)
    for i in range(config.request_count):
        task = task_order[i % len(task_order)]
        document_id = (document_offset + i) % doc_count
        session_id = f"session-{(session_offset + i) % session_count}"
        layout = "instruction_before_document" if i % 2 else "document_before_instruction"
        question = (
            f"Adversarial churn request {i}: switch to {task} on isolated document "
            f"{document_id} without relying on prior turns."
        )
        yield _regime_record(
            config,
            cache_config,
            i,
            task,
            document_for(document_id),
            document_id,
            session_id,
            question,
            layout,
        )


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
        document = row.get("document", "")
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
