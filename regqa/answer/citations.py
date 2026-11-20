"""Citation checking: every claim must point at a paragraph we actually retrieved.

An answer passes only if:
1. every sentence carries at least one citation like [§ 1005.6(b)(1)];
2. every cited section was in the retrieved sources, and a cited paragraph
   label (when given) exists in that section's retrieved paragraphs;
3. every figure in a sentence ($50, 60 days, 10 business days, 2%) appears in
   the text of a paragraph that sentence cites.

Rule 3 matters most in regulation answers: a model that writes "$500" where the
rule says "$50" produces an answer that *looks* cited but is wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from regqa.corpus.chunks import Chunk

CITE = re.compile(r"\[§\s*(\d+\.\d+[a-z]?)((?:\([A-Za-z0-9]+\))*)\]")
FIGURE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b)|\b\d+\s+(?:business\s+|calendar\s+|banking\s+)?(?:days?|months?|years?|hours?|minutes?)\b", re.I)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
SENTENCE = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z])")  # a citation after the period stays with its sentence


@dataclass
class CitationReport:
    ok: bool
    sentences: int
    cited_sentences: int
    unknown_citations: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    ungrounded_figures: list[str] = field(default_factory=list)
    cited_sections: list[str] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        out = []
        if self.uncited_sentences:
            out.append(f"{len(self.uncited_sentences)} sentence(s) have no citation: " + " | ".join(s[:80] for s in self.uncited_sentences[:3]))
        if self.unknown_citations:
            out.append("Citations not in the retrieved sources: " + ", ".join(self.unknown_citations))
        if self.ungrounded_figures:
            out.append("Figures not found in the cited text: " + ", ".join(self.ungrounded_figures))
        return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text.strip()) if len(s.strip()) > 3]


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(s)}


def check(answer: str, sources: list[Chunk]) -> CitationReport:
    by_section: dict[str, list[Chunk]] = {}
    for c in sources:
        by_section.setdefault(c.section, []).append(c)
    sentences = split_sentences(answer)
    report = CitationReport(ok=True, sentences=len(sentences), cited_sentences=0)
    for sent in sentences:
        cites = CITE.findall(sent)
        if not cites:
            report.uncited_sentences.append(sent)
            continue
        report.cited_sentences += 1
        cited_text = []
        for section, para in cites:
            label = f"{section}{para}"
            chunks = by_section.get(section)
            if not chunks:
                report.unknown_citations.append(f"§ {label}")
                continue
            if section not in report.cited_sections:
                report.cited_sections.append(section)
            if para:
                matching = [p for c in chunks for p in c.paragraphs if p.label == label or p.label.startswith(label + "(") or label.startswith(p.label)]
                if not matching:
                    report.unknown_citations.append(f"§ {label}")
                    continue
                cited_text.extend(p.text for p in matching)
            else:
                cited_text.extend(c.text for c in chunks)
        available = _numbers(" ".join(cited_text))
        for fig in FIGURE.findall(CITE.sub("", sent)):
            if not _numbers(fig) <= available:
                report.ungrounded_figures.append(fig.strip())
    report.ok = not (report.uncited_sentences or report.unknown_citations or report.ungrounded_figures)
    return report
