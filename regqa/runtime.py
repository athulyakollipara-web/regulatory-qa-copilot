"""Wires the corpus, retriever, model and checks together for the configured mode."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from regqa.config import Settings, get_settings
from regqa.corpus.chunks import Chunk, build_chunks
from regqa.corpus.ecfr import load
from regqa.retrieval.embeddings import HashingEmbedder, TitanEmbedder
from regqa.retrieval.hybrid import BedrockReranker, HybridRetriever, LexicalReranker
from regqa.safety import BedrockGuardrail, local_input_screen


@dataclass
class Runtime:
    settings: Settings
    chunks: list[Chunk]
    retriever: HybridRetriever
    graph: Any

    def ask(self, question: str) -> dict:
        s = self.graph.invoke({"question": question})
        hits = s.get("hits", [])
        return {
            "question": question,
            "answer": s.get("answer"),
            "found": bool(s.get("found")) and not s.get("blocked"),
            "mode": s.get("answer_mode"),
            "support": s.get("support"),
            "sources": [{"cite": f"§ {h.chunk.chunk_id.split('#')[0]}", "section": h.chunk.section, "heading": h.chunk.heading,
                         "regulation": h.chunk.regulation, "url": h.chunk.url, "score": h.score} for h in hits],
            "citations": s.get("citations"),
            "grounding": s.get("grounding"),
            "trace": s.get("trace", []),
        }


def build_runtime(settings: Settings | None = None, corpus: Path | None = None, llm=None) -> Runtime:
    from regqa.agent.graph import build_graph

    s = settings or get_settings()
    path = corpus or s.corpus
    if not path.exists():
        raise FileNotFoundError(f"No corpus at {path}. Build it with: python -m regqa.corpus.ecfr --out {path}")
    chunks = build_chunks(load(path))
    embedder = TitanEmbedder(s.embedding_model, s.region) if s.embeddings == "titan" else HashingEmbedder()
    reranker = BedrockReranker(s.rerank_model_arn, s.rerank_region) if s.reranker == "bedrock" else LexicalReranker()
    retriever = HybridRetriever(chunks, embedder, reranker)

    guardrail = BedrockGuardrail(s.guardrail_id, s.guardrail_version, s.region) if s.guardrail_id else None
    screen = guardrail.screen_input if guardrail else local_input_screen
    if llm is None and s.llm == "bedrock":
        from regqa.llm.bedrock import BedrockLLM

        llm = BedrockLLM(s.primary_model, s.fallback_model, s.region)
    graph = build_graph(retriever, screen, llm=llm, guardrail=guardrail, top_k=s.top_k, support_threshold=s.support_threshold)
    return Runtime(s, chunks, retriever, graph)


@lru_cache(maxsize=1)
def default_runtime() -> Runtime:
    return build_runtime()
