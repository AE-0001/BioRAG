from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import Document, SearchHit


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FaissStore:
    """Persistent cosine FAISS index; metadata remains separately auditable."""

    def __init__(self, embeddings: EmbeddingProvider):
        import faiss

        self.faiss = faiss
        self.embeddings = embeddings
        self.index = None
        self.documents: list[Document] = []

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        if hasattr(self.embeddings, "embed_evidence"):
            raw_vectors = self.embeddings.embed_evidence(documents)
        else:
            raw_vectors = self.embeddings.embed_documents(
                [f"{document.title}\n{document.text}" for document in documents]
            )
        vectors = np.asarray(raw_vectors, dtype="float32")
        self.faiss.normalize_L2(vectors)
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        if self.index.d != vectors.shape[1]:
            raise ValueError("Embedding dimensionality changed; rebuild the index")
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if self.index is None:
            return []
        query_vector = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        self.faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, min(top_k, len(self.documents)))
        return [
            SearchHit(
                document=self.documents[index],
                score=float(score),
                lexical_score=0.0,
                semantic_score=float(score),
            )
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self, directory: Path) -> None:
        if self.index is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(directory / "vectors.faiss"))
        (directory / "documents.json").write_text(
            json.dumps([document.to_dict() for document in self.documents], indent=2),
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "embedding_provider": getattr(
                        self.embeddings, "name", type(self.embeddings).__name__
                    ),
                    "dimensions": self.index.d,
                    "vectors": self.index.ntotal,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            configured = getattr(self.embeddings, "name", type(self.embeddings).__name__)
            if manifest.get("embedding_provider") != configured:
                raise ValueError(
                    "The saved FAISS index uses a different embedding provider; rebuild it."
                )
        self.index = self.faiss.read_index(str(directory / "vectors.faiss"))
        payload = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        self.documents = [Document.from_dict(item) for item in payload]
        if self.index.ntotal != len(self.documents):
            raise ValueError("FAISS vector/document count mismatch; rebuild the index")
