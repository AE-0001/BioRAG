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
from .local_embeddings import LocalHashEmbeddings
from .models import AgentTrace, Answer
from .retrieval import HybridIndex


def graph_state_to_answer(question: str, state: dict) -> Answer:
    """Normalize LangGraph's state dictionary to the response contract used by every interface."""
    return Answer(
        question=question,
        answer=state["answer"],
        citations=state.get("citations", []),
        evidence=state.get("hits", []),
        grounded=state.get("grounded", False),
        trace=[
            AgentTrace(
                agent=step["agent"],
                action=step["action"],
                detail=step.get("detail", "completed"),
            )
            for step in state.get("trace", [])
        ],
    )


class BioRAGService:
    def __init__(self, index_path: Path, production: bool = False):
        self.settings = Settings()
        self.production = production
        self.index_path = index_path
        self.index = HybridIndex.load(index_path) if index_path.exists() else HybridIndex()
        self.semantic = None
        self.retriever: Any = self.index
        if production:
            if self.settings.embedding_model == "local-hash":
                embeddings = LocalHashEmbeddings()
            else:
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
                try:
                    raw.extend(docling.ingest(path))
                except (ImportError, MemoryError, NotImplementedError, OSError, RuntimeError):
                    # OCR/model backends are optional and can be unavailable on
                    # a local CPU. Preserve a working, provenance-aware PDF path.
                    raw.extend(ingest_path(path))
            else:
                raw.extend(ingest_path(path))
        chunks = chunk_documents(raw)
        existing_ids = {document.id for document in self.index.documents}
        new_lexical_chunks = [document for document in chunks if document.id not in existing_ids]
        self.index.add(new_lexical_chunks)
        self.index.save(self.index_path)
        if self.semantic is not None:
            semantic_ids = {document.id for document in self.semantic.documents}
            new_semantic_chunks = [document for document in chunks if document.id not in semantic_ids]
            self.semantic.add(new_semantic_chunks)
            self.semantic.save(self.index_path.parent / "faiss")
        return {
            "documents": len({document.metadata.get("paper_id", document.source) for document in raw}),
            "chunks": len(new_lexical_chunks),
            "figures": sum(document.modality == "figure" for document in new_lexical_chunks),
            "tables": sum(document.metadata.get("kind") == "table" for document in new_lexical_chunks),
            "dataset_rows": sum(
                document.metadata.get("kind") == "supplementary_dataset"
                for document in new_lexical_chunks
            ),
            "indexed_total": len(self.index.documents),
        }

    def ask(self, question: str, top_k: int = 5):
        if self.production:
            generator = (
                None
                if self.settings.embedding_model == "local-hash"
                else GeminiGenerator(model=self.settings.generation_model)
            )
            state = FourAgentResearchGraph(self.retriever, generator).invoke(question, top_k)
            return graph_state_to_answer(question, state)
        return BioRAGGraph(self.index).ask(question, top_k)

    def search(self, query: str, top_k: int = 5) -> list:
        return self.retriever.search(query, top_k)

    def metrics(self) -> dict[str, int | str]:
        documents = self.index.documents
        return {
            "backend": (
                f"faiss+bm25+{self.semantic.embeddings.name}"
                if self.semantic is not None
                else "offline-test-double"
            ),
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
