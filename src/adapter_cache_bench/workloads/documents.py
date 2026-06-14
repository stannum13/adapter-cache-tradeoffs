from __future__ import annotations

PUBLIC_DOMAIN_SENTENCES = [
    "The observatory records pressure temperature wind and cloud cover every hour.",
    "A maintainer writes incident notes with symptoms actions and follow up owners.",
    "The river archive stores measurements by station year season and instrument.",
    "A causal transformer serving cluster receives repeated prompts over shared context.",
    "Adapter specialization can improve task accuracy while changing cache reuse patterns.",
    "The document includes tables bullet notes quoted policies and code-like fragments.",
]


def make_document(document_id: int, target_tokens: int = 180) -> str:
    tokens: list[str] = [f"document_{document_id}"]
    cursor = document_id
    while len(tokens) < target_tokens:
        sentence = PUBLIC_DOMAIN_SENTENCES[cursor % len(PUBLIC_DOMAIN_SENTENCES)]
        tokens.extend(sentence.split())
        tokens.append(f"fact_{document_id}_{cursor}")
        cursor += 1
    return " ".join(tokens[:target_tokens])
