"""Hybrid retrieval over regulation chunks: BM25 + vector k-NN -> Reciprocal Rank Fusion -> rerank.

Regulation questions mix exact legal terms ("access device", "adverse action")
that BM25 handles well with plain-English paraphrases ("lost debit card",
"loan was denied") that need vector similarity. RRF merges the two rankings by
position, so their score scales never need calibrating. A second-stage reranker
then reads the question and each candidate together to order the final top-k.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from regqa.corpus.chunks import Chunk
from regqa.retrieval.bm25 import BM25
from regqa.retrieval.embeddings import Embedder, cosine
from regqa.retrieval.glossary import expand
from regqa.retrieval.text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float  # reranker relevance, roughly 0..1
    sources: tuple[str, ...]


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class LexicalReranker:
    """Offline reranker: question-term coverage of the chunk, with extra weight on the section heading.

    Used locally and in CI. In AWS mode a hosted cross-encoder (``BedrockReranker``)
    replaces it through the same interface.
    """

    name = "lexical"

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        q = set(tokenize(query))
        out = []
        for h in hits:
            body = set(tokenize(h.chunk.text))
            head = set(tokenize(h.chunk.heading))
            coverage = len(q & body) / max(1, len(q))
            heading_hit = len(q & head) / max(1, len(q))
            score = 0.55 * coverage + 0.25 * heading_hit + 0.20 * h.score
            out.append(Hit(h.chunk, round(score, 4), h.sources))
        return sorted(out, key=lambda h: h.score, reverse=True)[:top_k]


class BedrockReranker:
    """Cross-encoder reranking with a Bedrock rerank model (e.g. Amazon Rerank 1.0).

    Rerank models are offered in a subset of regions; set ``RERANK_REGION`` and
    ``RERANK_MODEL_ARN`` to one enabled in your account.
    """

    name = "bedrock"

    def __init__(self, model_arn: str, region: str):
        import boto3

        self.client = boto3.client("bedrock-agent-runtime", region_name=region)
        self.model_arn = model_arn

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        resp = self.client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[{"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": h.chunk.search_text[:4000]}}}
                     for h in hits],
            rerankingConfiguration={"type": "BEDROCK_RERANKING_MODEL", "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": self.model_arn}, "numberOfResults": min(top_k, len(hits))}},
        )
        return [Hit(hits[r["index"]].chunk, round(float(r["relevanceScore"]), 4), hits[r["index"]].sources)
                for r in resp["results"]]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, reranker: Reranker | None = None, rrf_k: int = 60):
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = reranker or LexicalReranker()
        self.rrf_k = rrf_k
        texts = [c.search_text for c in chunks]
        self.bm25 = BM25(texts)
        # Headings are short and written in the regulation's own vocabulary, so they get their own ranking.
        self.heading_bm25 = BM25([f"{c.regulation} {c.heading}" for c in chunks])
        self.vectors = embedder.embed(texts)

    def _rank(self, scores: list[float], n: int) -> list[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:n] if scores[i] > 0]

    def candidates(self, query: str, n: int = 30) -> list[Hit]:
        query = expand(query)
        lexical = self._rank(self.bm25.scores(query), n)
        headings = self._rank(self.heading_bm25.scores(query), n)
        qv = self.embedder.embed([query])[0]
        semantic = self._rank([cosine(qv, v) for v in self.vectors], n)
        fused: dict[int, float] = {}
        sources: dict[int, set[str]] = {}
        for name, ranking in (("bm25", lexical), ("heading", headings), ("knn", semantic)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                sources.setdefault(idx, set()).add(name)
        top = max(fused.values(), default=1.0)
        hits = [Hit(self.chunks[i], s / top, tuple(sorted(sources[i]))) for i, s in fused.items()]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:n]

    def search(self, query: str, k: int = 5, n: int = 30) -> list[Hit]:
        return self.reranker.rerank(expand(query), self.candidates(query, n), k)
