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


def join_title_abstract(title: object, abstract: object) -> str:
    title_text = "" if title is None else str(title)
    abstract_text = "" if abstract is None else str(abstract)
    return f"{title_text} [SEP] {abstract_text}".strip()


def detect_genai(title: object, abstract: object) -> bool:
    text = f"{title or ''} {abstract or ''}".lower()
    return any(keyword in text for keyword in GENAI_KEYWORDS)

