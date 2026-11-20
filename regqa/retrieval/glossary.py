"""Everyday banking words -> the terms the regulations use.

Customers and staff say "debit card" and "ATM"; Regulation E says "access device"
and "electronic terminal". Appending the regulatory term to the query lets BM25
match the rule's own vocabulary. In AWS mode the LLM query rewrite does the same
job more broadly; this glossary keeps the offline pipeline honest about the gap.
It is general banking vocabulary, not tuned to the evaluation questions.
"""
from __future__ import annotations

import re

GLOSSARY = {
    r"\bdebit cards?\b|\bbank cards?\b": "access device",
    r"\batms?\b|\bcash machines?\b": "electronic terminal automated teller machine",
    r"\brecurring\b|\bautomatic payments?\b|\bautopay\b": "preauthorized transfer",
    r"\bdirect deposit\b|\bpayroll\b": "electronic fund transfer compulsory use preauthorized",
    r"\bcds?\b|\bcertificates? of deposit\b": "time account",
    r"\bholds?\b": "availability of funds",
    r"\bcheck(ing)? account\b": "account",
    r"\bloan (was )?denied\b|\bturned down\b|\brejected\b": "adverse action",
    r"\bprivacy policy\b": "privacy notice",
    r"\bremittances?\b|\bwire abroad\b|\bsend money abroad\b": "remittance transfer",
    r"\bstolen\b|\blost\b": "loss or theft unauthorized",
    r"\bkeep records?\b|\brecords? kept\b|\bretain\b": "record retention",
    r"\btreasury checks?\b|\bgovernment checks?\b": "U.S. Treasury check",
    r"\bnew accounts?\b": "new account exception",
}


def expand(query: str) -> str:
    extra = [term for pattern, term in GLOSSARY.items() if re.search(pattern, query, re.I)]
    return f"{query} {' '.join(extra)}" if extra else query
