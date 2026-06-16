from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASKS = ["qa", "json", "summary", "code"]
LAYOUTS = ["document_before_instruction", "instruction_before_document"]

SOURCES: list[dict[str, Any]] = [
    {
        "document_id": "declaration-1776",
        "source_title": "United States Declaration of Independence",
        "source_url": "https://www.archives.gov/founding-docs/declaration-transcript",
        "trust_group_id": "founding-docs",
        "document": (
            "We hold these truths to be self-evident that all men are created equal "
            "and endowed with unalienable Rights including Life Liberty and the "
            "pursuit of Happiness."
        ),
        "qa": ("Which three rights are named?", "Life Liberty and the pursuit of Happiness"),
        "json": (
            "Extract the named rights.",
            {"rights": ["Life", "Liberty", "pursuit of Happiness"]},
        ),
        "summary": (
            "Summarize the claim.",
            "The passage says people are equal and have rights including life and liberty.",
        ),
        "code": (
            "Write parser behavior checks for rights extraction.",
            {"tests": ["extracts_life", "extracts_liberty", "extracts_pursuit_of_happiness"]},
        ),
    },
    {
        "document_id": "gettysburg-1863",
        "source_title": "Gettysburg Address",
        "source_url": "https://www.abrahamlincolnonline.org/lincoln/speeches/gettysburg.htm",
        "trust_group_id": "civil-war-docs",
        "document": (
            "Four score and seven years ago our fathers brought forth on this "
            "continent a new nation conceived in Liberty and dedicated to the "
            "proposition that all men are created equal."
        ),
        "qa": ("What was the new nation conceived in?", "Liberty"),
        "json": (
            "Extract the founding value and proposition.",
            {"founding_value": "Liberty", "proposition": "all men are created equal"},
        ),
        "summary": (
            "Summarize the passage.",
            "The passage recalls a nation founded in liberty and committed to equality.",
        ),
        "code": (
            "Write parser checks for founding claims.",
            {"tests": ["captures_liberty_value", "captures_equality_proposition"]},
        ),
    },
    {
        "document_id": "constitution-preamble-1787",
        "source_title": "United States Constitution Preamble",
        "source_url": "https://www.archives.gov/founding-docs/constitution-transcript",
        "trust_group_id": "founding-docs",
        "document": (
            "We the People of the United States in Order to form a more perfect "
            "Union establish Justice insure domestic Tranquility and provide for "
            "the common defence."
        ),
        "qa": ("Who is named at the start of the passage?", "We the People"),
        "json": (
            "Extract two stated purposes.",
            {"purposes": ["form a more perfect Union", "establish Justice"]},
        ),
        "summary": (
            "Summarize the preamble excerpt.",
            "The excerpt states public purposes for union justice tranquility and defense.",
        ),
        "code": (
            "Write parser checks for preamble purposes.",
            {
                "tests": [
                    "captures_people_subject",
                    "captures_union_purpose",
                    "captures_justice_purpose",
                ]
            },
        ),
    },
    {
        "document_id": "federalist-10",
        "source_title": "Federalist No. 10",
        "source_url": "https://guides.loc.gov/federalist-papers/text-1-10",
        "trust_group_id": "founding-docs",
        "document": (
            "Among the advantages promised by a well constructed Union none "
            "deserves more accurate development than its tendency to break and "
            "control the violence of faction."
        ),
        "qa": ("What problem can the Union control?", "the violence of faction"),
        "json": (
            "Extract the structure and problem.",
            {"structure": "well constructed Union", "problem": "violence of faction"},
        ),
        "summary": (
            "Summarize the political claim.",
            "A well constructed Union is presented as a way to control factional violence.",
        ),
        "code": (
            "Write parser checks for the political claim.",
            {"tests": ["captures_union_structure", "captures_faction_problem"]},
        ),
    },
    {
        "document_id": "common-sense-1776",
        "source_title": "Common Sense",
        "source_url": "https://www.gutenberg.org/ebooks/147",
        "trust_group_id": "revolutionary-pamphlets",
        "document": (
            "Society is produced by our wants and government by our wickedness. "
            "The former promotes happiness positively by uniting our affections."
        ),
        "qa": ("What produces society according to the passage?", "our wants"),
        "json": (
            "Extract the source of society and government.",
            {"society_source": "our wants", "government_source": "our wickedness"},
        ),
        "summary": (
            "Summarize the contrast.",
            "The passage contrasts society from wants with government from wickedness.",
        ),
        "code": (
            "Write parser checks for the contrast.",
            {"tests": ["captures_society_source", "captures_government_source"]},
        ),
    },
    {
        "document_id": "alice-1865",
        "source_title": "Alice's Adventures in Wonderland",
        "source_url": "https://www.gutenberg.org/ebooks/11",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "Alice was beginning to get very tired of sitting by her sister on the "
            "bank and of having nothing to do. She peeped into the book her sister "
            "was reading."
        ),
        "qa": ("Who sat with Alice on the bank?", "her sister"),
        "json": (
            "Extract Alice's companion and location.",
            {"companion": "her sister", "location": "bank"},
        ),
        "summary": (
            "Summarize the scene.",
            "Alice is bored beside her sister on the bank and looks at her sister's book.",
        ),
        "code": (
            "Write parser checks for scene extraction.",
            {"tests": ["captures_alice", "captures_sister_companion", "captures_bank_location"]},
        ),
    },
    {
        "document_id": "frankenstein-1818",
        "source_title": "Frankenstein; or, The Modern Prometheus",
        "source_url": "https://www.gutenberg.org/ebooks/84",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "I am by birth a Genevese and my family is one of the most distinguished "
            "of that republic. My ancestors had been counsellors and syndics."
        ),
        "qa": ("Where is the narrator from by birth?", "Genevese"),
        "json": (
            "Extract birth identity and ancestor offices.",
            {"birth_identity": "Genevese", "ancestor_offices": ["counsellors", "syndics"]},
        ),
        "summary": (
            "Summarize the family background.",
            "The narrator is Genevese and comes from a distinguished civic family.",
        ),
        "code": (
            "Write parser checks for biographical extraction.",
            {"tests": ["captures_birth_identity", "captures_ancestor_offices"]},
        ),
    },
    {
        "document_id": "pride-prejudice-1813",
        "source_title": "Pride and Prejudice",
        "source_url": "https://www.gutenberg.org/ebooks/1342",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "It is a truth universally acknowledged that a single man in possession "
            "of a good fortune must be in want of a wife."
        ),
        "qa": ("What is the single man said to possess?", "a good fortune"),
        "json": (
            "Extract the man's status and supposed want.",
            {"status": "single man", "possession": "good fortune", "want": "wife"},
        ),
        "summary": (
            "Summarize the social claim.",
            "The passage states that a wealthy single man is presumed to want a wife.",
        ),
        "code": (
            "Write parser checks for the social claim.",
            {"tests": ["captures_single_man", "captures_good_fortune", "captures_wife_want"]},
        ),
    },
    {
        "document_id": "sherlock-1892",
        "source_title": "The Adventures of Sherlock Holmes",
        "source_url": "https://www.gutenberg.org/ebooks/1661",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "To Sherlock Holmes she is always the woman. I have seldom heard him "
            "mention her under any other name."
        ),
        "qa": ("How does Sherlock Holmes refer to her?", "the woman"),
        "json": (
            "Extract Holmes's phrase and narrator observation.",
            {"holmes_phrase": "the woman", "observation": "seldom used any other name"},
        ),
        "summary": (
            "Summarize Holmes's regard.",
            "Holmes regards her as uniquely important and usually calls her the woman.",
        ),
        "code": (
            "Write parser checks for character-reference extraction.",
            {"tests": ["captures_holmes_phrase", "captures_rare_other_name_observation"]},
        ),
    },
    {
        "document_id": "moby-dick-1851",
        "source_title": "Moby-Dick; or, The Whale",
        "source_url": "https://www.gutenberg.org/ebooks/2701",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "Call me Ishmael. Some years ago having little or no money in my purse "
            "and nothing particular to interest me on shore I thought I would sail about."
        ),
        "qa": ("What name does the narrator give?", "Ishmael"),
        "json": (
            "Extract the narrator name and reason for travel.",
            {"narrator": "Ishmael", "reason": "little or no money and little interest on shore"},
        ),
        "summary": (
            "Summarize the narrator's situation.",
            "Ishmael says lack of money and shore interest led him to sail.",
        ),
        "code": (
            "Write parser checks for narrator setup.",
            {
                "tests": [
                    "captures_ishmael_name",
                    "captures_money_shortage",
                    "captures_sailing_decision",
                ]
            },
        ),
    },
    {
        "document_id": "tale-two-cities-1859",
        "source_title": "A Tale of Two Cities",
        "source_url": "https://www.gutenberg.org/ebooks/98",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "It was the best of times it was the worst of times it was the age of "
            "wisdom it was the age of foolishness."
        ),
        "qa": ("Which two kinds of times are contrasted?", "best of times and worst of times"),
        "json": (
            "Extract the paired contrasts.",
            {"time_contrast": ["best", "worst"], "age_contrast": ["wisdom", "foolishness"]},
        ),
        "summary": (
            "Summarize the contrast.",
            "The passage frames the period through paired opposites of fortune and judgment.",
        ),
        "code": (
            "Write parser checks for parallel contrasts.",
            {"tests": ["captures_best_worst", "captures_wisdom_foolishness"]},
        ),
    },
    {
        "document_id": "wizard-oz-1900",
        "source_title": "The Wonderful Wizard of Oz",
        "source_url": "https://www.gutenberg.org/ebooks/55",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "Dorothy lived in the midst of the great Kansas prairies with Uncle Henry "
            "who was a farmer and Aunt Em who was the farmer's wife."
        ),
        "qa": ("Where did Dorothy live?", "the great Kansas prairies"),
        "json": (
            "Extract Dorothy's location and relatives.",
            {"location": "great Kansas prairies", "relatives": ["Uncle Henry", "Aunt Em"]},
        ),
        "summary": (
            "Summarize Dorothy's household.",
            "Dorothy lives on the Kansas prairies with Uncle Henry and Aunt Em.",
        ),
        "code": (
            "Write parser checks for household extraction.",
            {"tests": ["captures_dorothy", "captures_kansas_prairies", "captures_relatives"]},
        ),
    },
    {
        "document_id": "jane-eyre-1847",
        "source_title": "Jane Eyre",
        "source_url": "https://www.gutenberg.org/ebooks/1260",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "There was no possibility of taking a walk that day. The cold winter wind "
            "had brought with it clouds so sombre and a rain so penetrating."
        ),
        "qa": ("What prevented the walk?", "cold winter wind and penetrating rain"),
        "json": (
            "Extract the weather conditions.",
            {"wind": "cold winter wind", "rain": "penetrating rain", "clouds": "sombre"},
        ),
        "summary": (
            "Summarize the weather constraint.",
            "Bad winter weather with sombre clouds and penetrating rain prevented a walk.",
        ),
        "code": (
            "Write parser checks for weather extraction.",
            {
                "tests": [
                    "captures_winter_wind",
                    "captures_sombre_clouds",
                    "captures_penetrating_rain",
                ]
            },
        ),
    },
    {
        "document_id": "treasure-island-1883",
        "source_title": "Treasure Island",
        "source_url": "https://www.gutenberg.org/ebooks/120",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "Squire Trelawney Dr Livesey and the rest of these gentlemen having asked "
            "me to write down the whole particulars about Treasure Island."
        ),
        "qa": ("Who asked the narrator to write?", "Squire Trelawney and Dr Livesey"),
        "json": (
            "Extract the requesters and subject.",
            {"requesters": ["Squire Trelawney", "Dr Livesey"], "subject": "Treasure Island"},
        ),
        "summary": (
            "Summarize the narrator's assignment.",
            "The narrator says gentlemen asked him to record the details of Treasure Island.",
        ),
        "code": (
            "Write parser checks for request extraction.",
            {
                "tests": [
                    "captures_trelawney",
                    "captures_livesey",
                    "captures_treasure_island_subject",
                ]
            },
        ),
    },
    {
        "document_id": "wuthering-heights-1847",
        "source_title": "Wuthering Heights",
        "source_url": "https://www.gutenberg.org/ebooks/768",
        "trust_group_id": "gutenberg-fiction",
        "document": (
            "Wuthering Heights is the name of Mr Heathcliff's dwelling. Wuthering is "
            "a significant provincial adjective descriptive of atmospheric tumult."
        ),
        "qa": ("Whose dwelling is named Wuthering Heights?", "Mr Heathcliff"),
        "json": (
            "Extract the dwelling owner and word meaning.",
            {"dwelling_owner": "Mr Heathcliff", "wuthering_meaning": "atmospheric tumult"},
        ),
        "summary": (
            "Summarize the naming explanation.",
            "The passage identifies Heathcliff's dwelling and explains wuthering as stormy.",
        ),
        "code": (
            "Write parser checks for place-name extraction.",
            {"tests": ["captures_dwelling_owner", "captures_wuthering_definition"]},
        ),
    },
]


def build_records(repeats_per_layout: int = 2) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_index, source in enumerate(SOURCES):
        for task_type in TASKS:
            question, ground_truth = source[task_type]
            for layout in LAYOUTS:
                for repeat in range(repeats_per_layout):
                    record_id = (
                        f"source-expanded-{source['document_id']}-{task_type}-{layout}-r{repeat}"
                    )
                    records.append(
                        {
                            "request_id": record_id,
                            "session_id": f"source-expanded-session-{source_index:02d}-{repeat}",
                            "tenant_id": f"public-domain-{source_index % 3}",
                            "trust_group_id": source["trust_group_id"],
                            "document_id": source["document_id"],
                            "shared_prefix_id": source["document_id"],
                            "source_title": source["source_title"],
                            "source_url": source["source_url"],
                            "source_license": "public-domain",
                            "task_type": task_type,
                            "document": source["document"],
                            "question": f"{question} Return concise answer variant {repeat}.",
                            "ground_truth": ground_truth,
                            "expected_adapter": task_type,
                            "prompt_layout": layout,
                            "requires_json": task_type == "json",
                            "max_tokens": 64,
                        }
                    )
    return records


def write_dataset(output: str | Path, repeats_per_layout: int = 2) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in build_records(repeats_per_layout=repeats_per_layout):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/eval/source_eval_expanded.jsonl")
    parser.add_argument("--repeats-per-layout", type=int, default=2)
    args = parser.parse_args()
    print(write_dataset(args.output, repeats_per_layout=args.repeats_per_layout))


if __name__ == "__main__":
    main()
