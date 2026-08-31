"""Deterministic, dependency-free fake embeddings for tests and offline use.

The fake embeddings shipped in ``langchain_core`` (``FakeEmbeddings`` /
``DeterministicFakeEmbedding``) require numpy, which this project deliberately
does not install. This module provides a pure-Python equivalent:

- deterministic across processes and runs (SHA-256 feature hashing),
- normalized to unit length so cosine similarity is well-defined,
- similar texts share tokens and therefore get similar vectors.
"""

from __future__ import annotations

import hashlib
import math
import random
import re

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashEmbeddings(Embeddings, BaseModel):
    """Deterministic fake embeddings based on token hashing."""

    size: int = 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for token in _tokens(text):
            rng = random.Random(hashlib.sha256(token.encode("utf-8")).digest())
            index = rng.randrange(self.size)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
