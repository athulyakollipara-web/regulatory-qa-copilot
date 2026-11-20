"""Download and parse US banking regulations from the eCFR (Electronic Code of Federal Regulations).

The eCFR API is public and needs no key: https://www.ecfr.gov/developers/documentation/api/v1
US federal regulations are public-domain government works.

Default corpus (12 CFR, Banks and Banking):

    Part 1002  Regulation B   Equal Credit Opportunity
    Part 1005  Regulation E   Electronic Fund Transfers
    Part 1016  Regulation P   Privacy of Consumer Financial Information
    Part 1030  Regulation DD  Truth in Savings
    Part 229   Regulation CC  Availability of Funds and Collection of Checks

Usage (needs internet; runs in CI on GitHub's servers)::

    python -m regqa.corpus.ecfr --out data/corpus.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

API = "https://www.ecfr.gov/api/versioner/v1"
TITLE = 12
DEFAULT_PARTS = {"1002": "Regulation B", "1005": "Regulation E", "1016": "Regulation P", "1030": "Regulation DD", "229": "Regulation CC"}
USER_AGENT = "regulatory-qa-copilot (github.com/athulyakollipara-web/regulatory-qa-copilot)"

_MARKER = re.compile(r"^\s*\(([a-zA-Z]{1,4}|\d{1,3})\)")
_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"}


@dataclass
class Paragraph:
    label: str  # full citation label, e.g. "1005.6(b)(1)"
    text: str


@dataclass
class Section:
    part: str
    regulation: str
    section: str  # e.g. "1005.6"
    heading: str  # e.g. "Liability of consumer for unauthorized transfers."
    paragraphs: list[Paragraph] = field(default_factory=list)
    as_of: str = ""

    @property
    def url(self) -> str:
        return f"https://www.ecfr.gov/current/title-{TITLE}/section-{self.section}"


def _get(url: str, timeout: int = 120) -> bytes:
    # The eCFR API answers HTTP 406 to requests that don't accept a compressed response.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed public host
        body = resp.read()
        encoding = (resp.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip" or body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return body


def latest_date(title: int = TITLE) -> str:
    titles = json.loads(_get(f"{API}/titles.json"))["titles"]
    entry = next(t for t in titles if int(t["number"]) == title)
    return entry.get("up_to_date_as_of") or entry["latest_issue_date"]


_HEADING_THEN_MARKER = re.compile(r"^\s*(?:[^()]{0,160}?(?:\.|—|–|--|-|:))?\s*(?=\()")


def _next_letter(letter: str) -> str:
    return chr(ord(letter) + 1) if len(letter) == 1 and letter.isalpha() else ""


def _roman_successor(marker: str) -> str:
    order = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"]
    return order[order.index(marker) + 1] if marker in order[:-1] else ""


def _level(marker: str, stack: list[str], next_marker: str | None) -> int:
    """Nesting level of a paragraph marker: (a)=1, (1)=2, (i)=3, (A)=4.

    "(i)", "(v)" and "(x)" are both letters and roman numerals. In a definitions
    section, (i) follows (h); under (a)(1) it starts a roman list. When the marker is
    the letter that follows the open level-1 letter, the next paragraph decides:
    "(ii)" next means roman, anything else means letter.
    """
    if marker.isdigit():
        return 2
    if marker.isupper():
        return 4
    if marker in _ROMAN and stack:
        follows_letter = marker == _next_letter(stack[0])
        if follows_letter and next_marker != _roman_successor(marker):
            return 1  # (h) ... (i) ... (j): a letter
        return 3  # (a)(1) ... (i) (ii): a roman numeral
    return 1


def _markers(text: str) -> list[str]:
    """All markers a paragraph opens with: "(a) Heading —(1) Rule." -> ["a", "1"]."""
    out, rest = [], text
    while True:
        m = _MARKER.match(rest)
        if not m:
            return out
        out.append(m.group(1))
        rest = rest[m.end():]
        h = _HEADING_THEN_MARKER.match(rest)
        if not (h and _MARKER.match(rest[h.end():])):
            return out
        rest = rest[h.end():]


def label_paragraphs(section: str, texts: list[str]) -> list[Paragraph]:
    """Give every paragraph its full citation label by tracking (a)/(1)/(i)/(A) nesting."""
    texts = [re.sub(r"\s+", " ", t).strip() for t in texts]
    texts = [t for t in texts if t]
    markers = [_markers(t) for t in texts]
    stack: list[str] = []
    out: list[Paragraph] = []
    for i, (text, ms) in enumerate(zip(texts, markers)):
        following = next((m[0] for m in markers[i + 1:] if m), None)
        for j, marker in enumerate(ms):
            nxt = ms[j + 1] if j + 1 < len(ms) else following
            lvl = _level(marker, stack, nxt)
            stack = stack[: lvl - 1] + [marker]
        out.append(Paragraph(section + "".join(f"({m})" for m in stack), text))
    return out


def parse_part_xml(xml_bytes: bytes, part: str, regulation: str, as_of: str = "") -> list[Section]:
    root = ET.fromstring(xml_bytes)
    sections = []
    for div in root.iter():
        if not div.tag.startswith("DIV") or div.attrib.get("TYPE") != "SECTION":
            continue
        number = div.attrib.get("N", "").strip()
        head_el = div.find("HEAD")
        heading = " ".join(head_el.itertext()).strip() if head_el is not None else ""
        heading = re.sub(r"^§\s*[\d.]+\s*", "", heading).strip()
        texts = [" ".join(p.itertext()) for p in div if p.tag in ("P", "FP")]
        if not number or not texts or heading.lower().startswith("[reserved]"):
            continue
        sections.append(Section(part, regulation, number, heading, label_paragraphs(number, texts), as_of))
    return sections


def fetch(parts: dict[str, str] = DEFAULT_PARTS, date: str | None = None) -> list[Section]:
    date = date or latest_date()
    out: list[Section] = []
    for part, regulation in parts.items():
        xml_bytes = _get(f"{API}/full/{date}/title-{TITLE}.xml?part={part}")
        out.extend(parse_part_xml(xml_bytes, part, regulation, as_of=date))
    return out


def save(sections: list[Section], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for s in sections:
            fh.write(json.dumps({**asdict(s), "url": s.url}) + "\n")


def load(path: Path) -> list[Section]:
    sections = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            d.pop("url", None)
            d["paragraphs"] = [Paragraph(**p) for p in d["paragraphs"]]
            sections.append(Section(**d))
    return sections


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--date", default=None, help="eCFR point-in-time date (default: latest)")
    args = ap.parse_args()
    secs = fetch(date=args.date)
    save(secs, Path(args.out))
    n_par = sum(len(s.paragraphs) for s in secs)
    print(f"saved {len(secs)} sections / {n_par} paragraphs from {len(DEFAULT_PARTS)} parts to {args.out}")
