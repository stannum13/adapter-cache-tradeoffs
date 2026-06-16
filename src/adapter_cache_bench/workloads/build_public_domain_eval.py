from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DOCUMENTS = [
    {
        "document_id": "weather-log-1901",
        "document": (
            "weather-log-1901 station north records clear mornings rain at noon and a "
            "pressure drop before evening. observer ada notes that the river gauge "
            "stayed stable while wind shifted west."
        ),
        "qa": ("Who observed the station notes?", "ada"),
        "json": (
            "Extract observer and wind direction.",
            {"observer": "ada", "wind_direction": "west"},
        ),
        "summary": (
            "Summarize the weather log.",
            "Station north had clear mornings rain at noon a pressure drop and west wind.",
        ),
        "code": (
            "Write parser behavior checks for weather log rows.",
            {"tests": ["parses_observer_field", "captures_wind_direction"]},
        ),
    },
    {
        "document_id": "maintenance-note-1912",
        "document": (
            "maintenance-note-1912 describes a pump inspection. the belt was tightened "
            "the intake screen was cleaned and the operator scheduled a follow up check "
            "for monday."
        ),
        "qa": ("What day was the follow up scheduled?", "monday"),
        "json": (
            "Extract cleaned component and follow up day.",
            {"cleaned_component": "intake screen", "follow_up_day": "monday"},
        ),
        "summary": (
            "Summarize the maintenance note.",
            "The pump inspection tightened the belt cleaned the intake screen "
            "and set monday follow up.",
        ),
        "code": (
            "Write parser behavior checks for maintenance records.",
            {"tests": ["captures_follow_up_day", "handles_component_names"]},
        ),
    },
    {
        "document_id": "ledger-format-1915",
        "document": (
            "ledger-format-1915 uses lines with date item quantity and station separated "
            "by spaces. blank lines should be ignored and malformed lines should be reported."
        ),
        "qa": ("What should happen to blank lines?", "ignored"),
        "json": (
            "Extract the delimiter and malformed-line behavior.",
            {"delimiter": "spaces", "malformed_lines": "reported"},
        ),
        "summary": (
            "Summarize the ledger format.",
            "The ledger uses space separated date item quantity station rows "
            "and reports malformed lines.",
        ),
        "code": (
            "Write parser behavior checks for this format.",
            {"tests": ["ignores_blank_lines", "reports_malformed_lines"]},
        ),
    },
    {
        "document_id": "library-catalog-1920",
        "document": (
            "library-catalog-1920 lists title author shelf and condition. entries marked "
            "fragile require cotton gloves and the west archive room keeps the repair log."
        ),
        "qa": ("Where is the repair log kept?", "west archive room"),
        "json": (
            "Extract handling rule and repair log location.",
            {"handling_rule": "cotton gloves", "repair_log_location": "west archive room"},
        ),
        "summary": (
            "Summarize the catalog policy.",
            "Fragile catalog entries require cotton gloves and repairs are "
            "logged in the west archive room.",
        ),
        "code": (
            "Write parser behavior checks for catalog entries.",
            {"tests": ["detects_fragile_entries", "captures_shelf_field"]},
        ),
    },
    {
        "document_id": "harbor-shift-1924",
        "document": (
            "harbor-shift-1924 reports pier three closed at dusk after fog reduced "
            "visibility. clerk noor logged two delayed cargo carts and one cleared tug."
        ),
        "qa": ("Which pier closed at dusk?", "pier three"),
        "json": (
            "Extract clerk and delayed cargo carts.",
            {"clerk": "noor", "delayed_cargo_carts": 2},
        ),
        "summary": (
            "Summarize the harbor shift note.",
            "Fog reduced visibility pier three closed at dusk and noor logged delayed cargo carts.",
        ),
        "code": (
            "Write parser behavior checks for harbor shift notes.",
            {"tests": ["captures_pier_status", "parses_delayed_cart_count"]},
        ),
    },
]


TASKS = ["qa", "json", "summary", "code"]
LAYOUTS = ["document_before_instruction", "instruction_before_document"]


def build_records(count: int = 100) -> list[dict[str, Any]]:
    records = []
    for index in range(count):
        document = DOCUMENTS[index % len(DOCUMENTS)]
        task_type = TASKS[index % len(TASKS)]
        question, ground_truth = document[task_type]
        layout = LAYOUTS[(index // len(TASKS)) % len(LAYOUTS)]
        variant = index // (len(DOCUMENTS) * len(TASKS))
        document_id = f"{document['document_id']}-v{variant:03d}"
        variant_document = (
            f"{document['document']} audit-batch-{variant:03d} notes reference "
            f"marker public-domain-{variant % 17}."
        )
        records.append(
            {
                "request_id": f"eval-large-{index:04d}",
                "session_id": f"eval-session-{index % 10}",
                "tenant_id": f"eval-tenant-{index % 2}",
                "trust_group_id": f"eval-trust-{index % 2}",
                "document_id": document_id,
                "shared_prefix_id": document_id,
                "task_type": task_type,
                "document": variant_document,
                "question": f"{question} Use audit batch {variant:03d}.",
                "ground_truth": ground_truth,
                "expected_adapter": task_type,
                "prompt_layout": layout,
                "requires_json": task_type == "json",
                "max_tokens": 64,
            }
        )
    return records


def write_dataset(output: str | Path, count: int = 100) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in build_records(count):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/eval/public_domain_eval_large.jsonl")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    print(write_dataset(args.output, args.count))


if __name__ == "__main__":
    main()
