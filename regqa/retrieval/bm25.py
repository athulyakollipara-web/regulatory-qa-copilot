"""Okapi BM25 - the lexical half of hybrid retrieval.

In AWS mode OpenSearch does this natively; this implementation is used locally
and in tests so ranking behaviour can be checked without a cluster.
"""
from __future__ import annotations

import math
from collections import Counter

from regqa.retrieval.text import tokenize


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for term in q:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                s += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            out.append(s)
        return out
