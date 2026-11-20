"""Settings from environment variables.

* ``REGQA_LLM``      ``extractive`` (offline, no model) or ``bedrock``
* ``REGQA_RERANKER`` ``lexical`` (offline) or ``bedrock`` (hosted cross-encoder)
* ``REGQA_CORPUS``   path to the corpus JSONL built by ``python -m regqa.corpus.ecfr``
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    llm: str
    reranker: str
    embeddings: str
    corpus: Path
    region: str
    primary_model: str
    fallback_model: str
    embedding_model: str
    rerank_model_arn: str
    rerank_region: str
    guardrail_id: str | None
    guardrail_version: str
    top_k: int
    support_threshold: float


def get_settings() -> Settings:
    reranker = os.getenv("REGQA_RERANKER", "lexical")
    default_threshold = {"lexical": 0.42, "bedrock": 0.15}.get(reranker, 0.42)
    rerank_region = os.getenv("RERANK_REGION", "us-west-2")
    return Settings(
        llm=os.getenv("REGQA_LLM", "extractive"),
        reranker=reranker,
        embeddings=os.getenv("REGQA_EMBEDDINGS", "hashing"),
        corpus=Path(os.getenv("REGQA_CORPUS", ROOT / "data" / "corpus.jsonl")),
        region=os.getenv("AWS_REGION", "us-east-1"),
        # Override with the newest model IDs enabled in your Bedrock account/region.
        primary_model=os.getenv("BEDROCK_PRIMARY_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        fallback_model=os.getenv("BEDROCK_FALLBACK_MODEL", "us.amazon.nova-pro-v1:0"),
        embedding_model=os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"),
        rerank_model_arn=os.getenv("RERANK_MODEL_ARN", f"arn:aws:bedrock:{rerank_region}::foundation-model/amazon.rerank-v1:0"),
        rerank_region=rerank_region,
        guardrail_id=os.getenv("BEDROCK_GUARDRAIL_ID"),
        guardrail_version=os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
        top_k=int(os.getenv("REGQA_TOP_K", "5")),
        support_threshold=float(os.getenv("REGQA_SUPPORT_THRESHOLD", default_threshold)),
    )
