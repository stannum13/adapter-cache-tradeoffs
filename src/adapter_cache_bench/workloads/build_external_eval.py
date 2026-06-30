from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adapter_cache_bench.workloads.build_source_eval_expanded import SOURCES as BASE_SOURCES
from adapter_cache_bench.workloads.build_source_eval_expanded import TASKS

LAYOUTS = ["document_before_instruction", "instruction_before_document"]

EXTERNAL_SOURCES: list[dict[str, Any]] = [
    {
        "document_id": "walden-1854",
        "source_title": "Walden",
        "source_url": "https://www.gutenberg.org/ebooks/205",
        "trust_group_id": "gutenberg-nonfiction",
        "document": (
            "When I wrote the following pages I lived alone in the woods a mile from "
            "any neighbor in a house which I had built myself on the shore of Walden Pond."
        ),
        "qa": ("Where did the narrator live?", "alone in the woods"),
        "json": (
            "Extract residence and water body.",
            {"residence": "house built by the narrator", "water_body": "Walden Pond"},
        ),
        "summary": (
            "Summarize the living arrangement.",
            "The narrator lived alone in a self-built house near Walden Pond.",
        ),
        "code": (
            "Write parser checks for residence extraction.",
            {"tests": ["captures_alone_in_woods", "captures_self_built_house"]},
        ),
    },
    {
        "document_id": "douglass-1845",
        "source_title": "Narrative of the Life of Frederick Douglass",
        "source_url": "https://www.gutenberg.org/ebooks/23",
        "trust_group_id": "gutenberg-nonfiction",
        "document": (
            "I was born in Tuckahoe near Hillsborough and about twelve miles from "
            "Easton in Talbot county Maryland. I have no accurate knowledge of my age."
        ),
        "qa": ("Where was the narrator born?", "Tuckahoe"),
        "json": (
            "Extract birthplace and uncertainty.",
            {"birthplace": "Tuckahoe", "uncertainty": "no accurate knowledge of age"},
        ),
        "summary": (
            "Summarize the autobiographical detail.",
            "The narrator gives his birthplace and says he does not know his exact age.",
        ),
        "code": (
            "Write parser checks for autobiographical extraction.",
            {"tests": ["captures_tuckahoe_birthplace", "captures_age_uncertainty"]},
        ),
    },
    {
        "document_id": "modest-proposal-1729",
        "source_title": "A Modest Proposal",
        "source_url": "https://www.gutenberg.org/ebooks/1080",
        "trust_group_id": "gutenberg-essays",
        "document": (
            "It is a melancholy object to those who walk through this great town or "
            "travel in the country when they see the streets crowded with beggars."
        ),
        "qa": ("What sight is called melancholy?", "streets crowded with beggars"),
        "json": (
            "Extract setting and observed problem.",
            {"setting": "town and country", "problem": "streets crowded with beggars"},
        ),
        "summary": (
            "Summarize the social observation.",
            "The passage observes public poverty visible in crowded streets of beggars.",
        ),
        "code": (
            "Write parser checks for social-observation extraction.",
            {"tests": ["captures_melancholy_object", "captures_beggars"]},
        ),
    },
    {
        "document_id": "time-machine-1895",
        "source_title": "The Time Machine",
        "source_url": "https://www.gutenberg.org/ebooks/35",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "The Time Traveller for so it will be convenient to speak of him was "
            "expounding a recondite matter to us. His grey eyes shone and twinkled."
        ),
        "qa": ("Who was expounding a matter?", "The Time Traveller"),
        "json": (
            "Extract speaker and eye description.",
            {"speaker": "The Time Traveller", "eyes": "grey eyes shone and twinkled"},
        ),
        "summary": (
            "Summarize the introduction.",
            "The Time Traveller explains a difficult matter while his eyes shine.",
        ),
        "code": (
            "Write parser checks for character introduction.",
            {"tests": ["captures_time_traveller", "captures_eye_description"]},
        ),
    },
    {
        "document_id": "dracula-1897",
        "source_title": "Dracula",
        "source_url": "https://www.gutenberg.org/ebooks/345",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "Jonathan Harker's Journal kept in shorthand records that he left Munich "
            "at 8:35 P.M. on 1 May and arrived at Vienna early next morning."
        ),
        "qa": ("Who kept the journal?", "Jonathan Harker"),
        "json": (
            "Extract traveler and departure city.",
            {"traveler": "Jonathan Harker", "departure_city": "Munich"},
        ),
        "summary": (
            "Summarize the travel log.",
            "Jonathan Harker records leaving Munich and reaching Vienna the next morning.",
        ),
        "code": (
            "Write parser checks for travel-log extraction.",
            {"tests": ["captures_harker", "captures_munich", "captures_vienna"]},
        ),
    },
    {
        "document_id": "secret-garden-1911",
        "source_title": "The Secret Garden",
        "source_url": "https://www.gutenberg.org/ebooks/113",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "When Mary Lennox was sent to Misselthwaite Manor to live with her uncle "
            "everybody said she was the most disagreeable-looking child ever seen."
        ),
        "qa": ("Where was Mary Lennox sent?", "Misselthwaite Manor"),
        "json": (
            "Extract child and destination.",
            {"child": "Mary Lennox", "destination": "Misselthwaite Manor"},
        ),
        "summary": (
            "Summarize Mary's relocation.",
            "Mary Lennox is sent to live with her uncle at Misselthwaite Manor.",
        ),
        "code": (
            "Write parser checks for relocation extraction.",
            {"tests": ["captures_mary_lennox", "captures_misselthwaite_manor"]},
        ),
    },
    {
        "document_id": "peter-pan-1911",
        "source_title": "Peter Pan",
        "source_url": "https://www.gutenberg.org/ebooks/16",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "All children except one grow up. They soon know that they will grow up "
            "and the way Wendy knew was this."
        ),
        "qa": ("Who is named in the passage?", "Wendy"),
        "json": (
            "Extract the rule and named child.",
            {"rule": "all children except one grow up", "named_child": "Wendy"},
        ),
        "summary": (
            "Summarize the opening claim.",
            "The passage states that nearly all children grow up and introduces Wendy.",
        ),
        "code": (
            "Write parser checks for opening-rule extraction.",
            {"tests": ["captures_grow_up_rule", "captures_wendy"]},
        ),
    },
    {
        "document_id": "odyssey",
        "source_title": "The Odyssey",
        "source_url": "https://www.gutenberg.org/ebooks/1727",
        "trust_group_id": "gutenberg-classics",
        "document": (
            "Tell me O Muse of that ingenious hero who travelled far and wide after "
            "he had sacked the famous town of Troy."
        ),
        "qa": ("Who is asked to tell the story?", "Muse"),
        "json": (
            "Extract invoked figure and conquered city.",
            {"invoked_figure": "Muse", "city": "Troy"},
        ),
        "summary": (
            "Summarize the invocation.",
            "The speaker asks the Muse to tell of a hero who travelled after Troy fell.",
        ),
        "code": (
            "Write parser checks for epic invocation extraction.",
            {"tests": ["captures_muse_invocation", "captures_troy"]},
        ),
    },
    {
        "document_id": "republic",
        "source_title": "The Republic",
        "source_url": "https://www.gutenberg.org/ebooks/1497",
        "trust_group_id": "gutenberg-philosophy",
        "document": (
            "I went down yesterday to the Piraeus with Glaucon the son of Ariston "
            "that I might offer up my prayers to the goddess."
        ),
        "qa": ("Where did the narrator go?", "the Piraeus"),
        "json": (
            "Extract destination and companion.",
            {"destination": "Piraeus", "companion": "Glaucon"},
        ),
        "summary": (
            "Summarize the journey.",
            "The narrator went to the Piraeus with Glaucon to offer prayers.",
        ),
        "code": (
            "Write parser checks for journey extraction.",
            {"tests": ["captures_piraeus", "captures_glaucon", "captures_prayers"]},
        ),
    },
    {
        "document_id": "souls-black-folk-1903",
        "source_title": "The Souls of Black Folk",
        "source_url": "https://www.gutenberg.org/ebooks/408",
        "trust_group_id": "gutenberg-nonfiction",
        "document": (
            "Between me and the other world there is ever an unasked question. They "
            "approach me in a half-hesitant sort of way and eye me curiously."
        ),
        "qa": ("What stands between the narrator and the other world?", "an unasked question"),
        "json": (
            "Extract barrier and manner of approach.",
            {"barrier": "unasked question", "approach": "half-hesitant"},
        ),
        "summary": (
            "Summarize the social distance.",
            "The narrator describes a persistent unasked question separating him from others.",
        ),
        "code": (
            "Write parser checks for social-distance extraction.",
            {"tests": ["captures_unasked_question", "captures_hesitant_approach"]},
        ),
    },
]


def build_records(variants_per_task: int = 5) -> list[dict[str, Any]]:
    sources = [*BASE_SOURCES, *EXTERNAL_SOURCES]
    records: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        for task_type in TASKS:
            question, ground_truth = source[task_type]
            for variant in range(variants_per_task):
                layout = LAYOUTS[variant % len(LAYOUTS)]
                records.append(
                    {
                        "request_id": (
                            f"external-public-domain-{source['document_id']}-{task_type}-v{variant}"
                        ),
                        "session_id": f"external-session-{source_index:02d}-{variant}",
                        "tenant_id": f"external-tenant-{source_index % 5}",
                        "trust_group_id": source["trust_group_id"],
                        "document_id": source["document_id"],
                        "shared_prefix_id": source["document_id"],
                        "source_title": source["source_title"],
                        "source_url": source["source_url"],
                        "source_license": "public-domain",
                        "task_type": task_type,
                        "document": source["document"],
                        "question": f"{question} Use variant {variant}.",
                        "ground_truth": ground_truth,
                        "expected_adapter": task_type,
                        "prompt_layout": layout,
                        "requires_json": task_type == "json",
                        "max_tokens": 64,
                    }
                )
    return records


def write_dataset(output: str | Path, variants_per_task: int = 5) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in build_records(variants_per_task=variants_per_task):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/eval/external_public_domain_eval.jsonl")
    parser.add_argument("--variants-per-task", type=int, default=5)
    args = parser.parse_args()
    print(write_dataset(args.output, variants_per_task=args.variants_per_task))


if __name__ == "__main__":
    main()
