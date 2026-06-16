from __future__ import annotations


def ground_truth_for(task_type: str, document_id: int, request_index: int):
    if task_type == "json":
        return {"document_id": document_id, "field": f"fact_{document_id}_{request_index}"}
    if task_type == "code":
        return {"tests": ["parses_fact_ids", "handles_empty_lines"]}
    if task_type == "summary":
        return f"Summary for document {document_id}"
    return f"fact_{document_id}_{request_index}"
