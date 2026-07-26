from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents import BioRAGGraph
from .chunking import chunk_documents
from .config import Settings
from .docling_ingestion import DoclingPaperIngestor
from .faiss_store import FaissStore
from .gemini import GeminiEmbeddings, GeminiGenerator
from .graph import FourAgentResearchGraph
from .hybrid import ReciprocalRankFusion
from .ingestion import ingest_path
from .retrieval import HybridIndex


class BioRAGService:
    def __init__(self, index_path: Path, production: bool = False):
        self.settings = Settings()
        self.production = production
        self.index_path = index_path
        self.index = HybridIndex.load(index_path) if index_path.exists() else HybridIndex()
        self.semantic = None
        self.retriever: Any = self.index
        if production:
            embeddings = GeminiEmbeddings(model=self.settings.embedding_model)
            self.semantic = FaissStore(embeddings)
            faiss_dir = index_path.parent / "faiss"
            if (faiss_dir / "vectors.faiss").exists():
                self.semantic.load(faiss_dir)
            self.retriever = ReciprocalRankFusion(self.index, self.semantic)

    def ingest(self, paths: list[Path]) -> dict[str, int]:
        raw = []
        docling = DoclingPaperIngestor(self.settings.data_dir / "artifacts")
        for path in paths:
            if path.suffix.lower() == ".pdf" and self.settings.extraction_engine == "docling":
                raw.extend(docling.ingest(path))
            else:
                raw.extend(ingest_path(path))
        chunks = chunk_documents(raw)
        existing_ids = {document.id for document in self.index.documents}
        chunks = [document for document in chunks if document.id not in existing_ids]
        self.index.add(chunks)
        self.index.save(self.index_path)
        if self.semantic is not None:
            self.semantic.add(chunks)
            self.semantic.save(self.index_path.parent / "faiss")
        return {
            "documents": len({document.metadata.get("paper_id", document.source) for document in raw}),
            "chunks": len(chunks),
            "figures": sum(document.modality == "figure" for document in chunks),
            "tables": sum(document.metadata.get("kind") == "table" for document in chunks),
            "dataset_rows": sum(
                document.metadata.get("kind") == "supplementary_dataset" for document in chunks
            ),
            "indexed_total": len(self.index.documents),
        }

    def ask(self, question: str, top_k: int = 5):
        if self.production:
            generator = GeminiGenerator(model=self.settings.generation_model)
            return FourAgentResearchGraph(self.retriever, generator).invoke(question, top_k)
        return BioRAGGraph(self.index).ask(question, top_k)

    def search(self, query: str, top_k: int = 5) -> list:
        return self.retriever.search(query, top_k)

    def metrics(self) -> dict[str, int | str]:
        documents = self.index.documents
        return {
            "backend": "faiss+bm25+gemini" if self.production else "offline-test-double",
            "papers": len(
                {
                    document.metadata.get("paper_id", document.source)
                    for document in documents
                    if document.metadata.get("kind") == "paper_text"
                }
            ),
            "figures": sum(document.modality == "figure" for document in documents),
            "tables": sum(document.metadata.get("kind") == "table" for document in documents),
            "supplementary_rows": sum(
                document.metadata.get("kind") == "supplementary_dataset"
                for document in documents
            ),
            "chunks": len(documents),
        }
