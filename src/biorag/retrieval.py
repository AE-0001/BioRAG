from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from .models import Document, SearchHit

TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9-]{1,}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN.findall(text)]


class HashEmbedding:
    """Dependency-free signed feature hashing baseline for reproducible demos."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class HybridIndex:
    """BM25 + dense-vector retrieval with transparent score fusion."""

    def __init__(self, semantic_weight: float = 0.55):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        self.semantic_weight = semantic_weight
        self.embedder = HashEmbedding()
        self.documents: list[Document] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.embeddings: list[list[float]] = []
        self.average_length = 0.0

    def add(self, documents: list[Document]) -> None:
        for document in documents:
            terms = Counter(tokenize(f"{document.title} {document.text}"))
            self.documents.append(document)
            self.term_frequencies.append(terms)
            self.document_frequencies.update(terms.keys())
            self.embeddings.append(self.embedder.encode(f"{document.title} {document.text}"))
        total = sum(sum(terms.values()) for terms in self.term_frequencies)
        self.average_length = total / len(self.documents) if self.documents else 0.0

    def _bm25(self, query_terms: list[str], index: int, k1: float = 1.5, b: float = 0.75) -> float:
        frequencies = self.term_frequencies[index]
        length = sum(frequencies.values())
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            containing = self.document_frequencies[term]
            idf = math.log(1 + (len(self.documents) - containing + 0.5) / (containing + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * length / (self.average_length or 1.0)
            )
            score += idf * frequency * (k1 + 1) / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip() or not self.documents:
            return []
        query_terms = tokenize(query)
        query_vector = self.embedder.encode(query)
        lexical = [self._bm25(query_terms, i) for i in range(len(self.documents))]
        semantic = [max(0.0, cosine(query_vector, vector)) for vector in self.embeddings]
        max_lexical = max(lexical) or 1.0
        hits = [
            SearchHit(
                document=document,
                lexical_score=lexical[i] / max_lexical,
                semantic_score=semantic[i],
                score=(1 - self.semantic_weight) * lexical[i] / max_lexical
                + self.semantic_weight * semantic[i],
            )
            for i, document in enumerate(self.documents)
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "semantic_weight": self.semantic_weight,
                    "documents": [document.to_dict() for document in self.documents],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> HybridIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls(semantic_weight=payload["semantic_weight"])
        index.add([Document.from_dict(item) for item in payload["documents"]])
        return index
