from __future__ import annotations

GENAI_KEYWORDS = [
    "large language model",
    "llm",
    "chatgpt",
    "gpt-3",
    "gpt-4",
    "instruction tuning",
    "rlhf",
    "retrieval augmented generation",
    "rag",
    "diffusion model",
    "generative ai",
    "generative artificial intelligence",
    "foundation model",
    "prompting",
    "in-context learning",
    "chain-of-thought",
    "text-to-image",
]

# Keyword matching is intentionally simple and transparent. It is used as a
# coarse topic flag, not as a classifier for whether a paper truly studies GenAI.


def join_title_abstract(title: object, abstract: object) -> str:
    """Combine title and abstract text for embedding.

    Missing values are converted to empty strings so encoder inputs remain
    strings. The `[SEP]` marker preserves a weak boundary between title and
    abstract for transformer tokenization.
    """
    title_text = "" if title is None else str(title)
    abstract_text = "" if abstract is None else str(abstract)
    return f"{title_text} [SEP] {abstract_text}".strip()


def detect_genai(title: object, abstract: object) -> bool:
    """Detect whether title/abstract text contains a GenAI keyword.

    This is a case-insensitive substring check. It can miss synonyms and can
    produce false positives for ambiguous terms such as `rag`, so callers should
    treat the result as a reproducible heuristic rather than ground truth.
    """
    text = f"{title or ''} {abstract or ''}".lower()
    return any(keyword in text for keyword in GENAI_KEYWORDS)
