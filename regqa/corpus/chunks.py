"""Turn sections into retrieval chunks without losing paragraph-level citations.

A chunk is one top-level paragraph, e.g. § 1005.6(b), together with its nested
sub-paragraphs (b)(1), (b)(2), (b)(2)(i)... It is large enough to carry a
complete rule, small enough to rank precisely, and it keeps every sub-paragraph's
label so an answer can cite § 1005.6(b)(2) exactly and the citation can be checked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from regqa.corpus.ecfr import Paragraph, Section

MAX_CHARS = 2400


@dataclass
class Chunk:
    chunk_id: str  # e.g. "1005.6(b)"
    section: str
    heading: str
    regulation: str
    url: str
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.paragraphs)

    @property
    def search_text(self) -> str:
        """What retrieval indexes: regulation + section heading give context a bare paragraph lacks."""
        return f"{self.regulation} § {self.section} {self.heading} {self.text}"

    @property
    def labels(self) -> set[str]:
        return {p.label for p in self.paragraphs}


def _top(label: str) -> str:
    m = re.match(r"^([\d.]+[a-z]?(?:\([^)]+\))?)", label)
    return m.group(1) if m else label


def build_chunks(sections: list[Section]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s in sections:
        groups: dict[str, list[Paragraph]] = {}
        for p in s.paragraphs:
            groups.setdefault(_top(p.label), []).append(p)
        for top, paras in groups.items():
            # Very long top-level paragraphs are split, keeping each piece's labels.
            piece: list[Paragraph] = []
            size = 0
            part = 0
            for p in paras:
                if piece and size + len(p.text) > MAX_CHARS:
                    chunks.append(Chunk(f"{top}#{part}" if part else top, s.section, s.heading, s.regulation, s.url, piece))
                    piece, size, part = [], 0, part + 1
                piece.append(p)
                size += len(p.text)
            if piece:
                chunks.append(Chunk(f"{top}#{part}" if part else top, s.section, s.heading, s.regulation, s.url, piece))
    return chunks
