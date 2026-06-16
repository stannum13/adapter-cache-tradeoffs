from __future__ import annotations

from adapter_cache_bench.config import CacheConfig

TASK_INSTRUCTIONS = {
    "qa": "Answer the question using only the document.",
    "json": "Extract the requested fields as valid JSON.",
    "summary": "Write a concise factual summary.",
    "code": "Write a small parser or documentation helper for the document.",
}


def prompt_for(
    task_type: str,
    document: str,
    question: str,
    layout: str,
    adapter_id: str,
    use_invocation: bool = True,
    cache_config: CacheConfig | None = None,
) -> str:
    config = cache_config or CacheConfig()
    marker = config.invocation_markers.get(adapter_id, f"<ADAPTER:{adapter_id}>")
    invocation = f"{marker} " if use_invocation else ""
    instruction = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["qa"])
    task = f"{invocation}{instruction} Task: {question}"
    if layout == "document_before_instruction":
        return f"Document: {document} {task}"
    return f"{task} Document: {document}"
