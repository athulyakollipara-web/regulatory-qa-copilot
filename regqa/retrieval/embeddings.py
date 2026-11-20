"""Embedding models behind one small interface.

* ``TitanEmbedder``   - Amazon Titan Text Embeddings V2 on Bedrock (AWS mode).
* ``HashingEmbedder`` - deterministic feature-hashing embedder (word + character
  trigrams). No model download, no network; used locally and in CI. It captures
  surface similarity (e.g. "deposit" vs "deposits"), not deep semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Protocol

from regqa.retrieval.text import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for tok in tokenize(text):
                idx, sign = self._bucket("w:" + tok)
                v[idx] += 2.0 * sign
                padded = f"#{tok}#"
                for i in range(len(padded) - 2):
                    idx, sign = self._bucket("c:" + padded[i:i + 3])
                    v[idx] += sign
            out.append(_normalize(v))
        return out


class TitanEmbedder:
    def __init__(self, model_id: str, region: str, dim: int = 512):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True}),
            )
            vectors.append(json.loads(resp["body"].read())["embedding"])
        return vectors


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
