"""Dependency-free local embeddings for reproducible FAISS smoke tests."""

from __future__ import annotations

import hashlib
import re

TOKEN = re.compile(r"[A-Za-z0-9_]+")


class LocalHashEmbeddings:
    """Deterministic feature-hash vectors; useful when an API is unavailable."""

    name = "local-hash"

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_evidence(self, documents) -> list[list[float]]:
        return self.embed_documents([f"{document.title}\n{document.text}" for document in documents])
