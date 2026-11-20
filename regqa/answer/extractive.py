"""Offline answerer: quotes the most relevant regulation sentences, each with its citation.

It needs no model, so the whole system runs and is tested without AWS. It is
also the safe fallback when a generated answer fails the citation check: a
quoted, cited sentence from the rule is always better than an unverified one.
"""
from __future__ import annotations

import re

from regqa.retrieval.hybrid import Hit
from regqa.retrieval.text import tokenize

_SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
_LEAD = re.compile(r"^\s*(?:\([A-Za-z0-9]+\)\s*)+(?:[A-Z][^.]{0,60}\.\s+)?")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) > 20]


def answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    q = set(tokenize(question))
    scored = []
    for rank, h in enumerate(hits[:3]):
        for p in h.chunk.paragraphs:
            for s in _sentences(p.text):
                clean = _LEAD.sub("", s)
                overlap = len(q & set(tokenize(clean))) / max(1, len(q))
                scored.append((overlap - 0.05 * rank, clean, p.label))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked, seen = [], set()
    best = scored[0][0] if scored else 0
    for score, sent, label in scored:
        if score <= 0 or score < 0.6 * best or sent in seen:  # keep only sentences close to the best match
            continue
        seen.add(sent)
        sent = sent.rstrip(" ;.") + "."
        picked.append(f"{sent[0].upper()}{sent[1:]} [§ {label}]")
        if len(picked) == max_sentences:
            break
    return " ".join(picked)
