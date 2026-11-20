from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "of", "in", "for", "and", "or", "to", "is", "are", "by", "with", "what", "which", "how",
    "does", "do", "did", "was", "were", "be", "been", "on", "at", "as", "it", "its", "this", "that", "these",
    "those", "can", "may", "must", "shall", "should", "if", "when", "under", "any", "such", "than", "there",
    "their", "they", "from", "into", "about", "who", "whom", "not", "no", "my", "i", "we", "you", "your",
}


def stem(token: str) -> str:
    """Very light suffix stripping so 'transfers'/'transfer' and 'disclosures'/'disclosure' match."""
    for suffix in ("ies", "es", "s"):
        if len(token) > 4 and token.endswith(suffix) and not token.endswith("ss"):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
