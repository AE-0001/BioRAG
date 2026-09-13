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
from .local_ai import OllamaEmbeddings, OllamaGenerator
from .local_embeddings import LocalHashEmbeddings
from .markitdown_ingestion import MarkItDownIngestor
from .models import AgentTrace, Answer
from .observability import (
    INDEX_CHUNKS,
    INGEST_SECONDS,
    PARSER_TOTAL,
    QUERY_SECONDS,
    QUERY_TOTAL,
    observe,
)
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
            embeddings = self._build_embeddings()
            self.semantic = FaissStore(embeddings)
            faiss_dir = index_path.parent / "faiss"
            if (faiss_dir / "vectors.faiss").exists():
                self.semantic.load(faiss_dir)
            self.retriever = ReciprocalRankFusion(self.index, self.semantic)
        self.generator = self._build_generator() if production else None
        self.graph = (
            FourAgentResearchGraph(
                self.retriever,
                self.generator,
                minimum_relevance=getattr(self.settings, "minimum_relevance", 0.20),
                max_retrieval_attempts=getattr(self.settings, "max_retrieval_attempts", 2),
            )
            if production
            else None
        )

    def _build_embeddings(self):
        provider = getattr(
            self.settings,
            "embedding_provider",
            "local-hash" if self.settings.embedding_model == "local-hash" else "gemini",
        ).lower()
        if provider == "gemini":
            return GeminiEmbeddings(model=self.settings.embedding_model)
        if provider == "ollama":
            return OllamaEmbeddings(
                model=getattr(self.settings, "ollama_embedding_model", "nomic-embed-text"),
                base_url=getattr(self.settings, "ollama_base_url", "http://localhost:11434"),
                timeout=getattr(self.settings, "request_timeout", 60),
            )
        if provider == "local-hash":
            return LocalHashEmbeddings()
        raise ValueError(f"Unsupported embedding provider: {provider}")

    def _build_generator(self):
        provider = getattr(
            self.settings,
            "generation_provider",
            "none" if self.settings.embedding_model == "local-hash" else "gemini",
        ).lower()
        if provider == "gemini":
            return GeminiGenerator(model=self.settings.generation_model)
        if provider == "ollama":
            return OllamaGenerator(
                model=self.settings.generation_model,
                base_url=getattr(self.settings, "ollama_base_url", "http://localhost:11434"),
                timeout=getattr(self.settings, "request_timeout", 60),
            )
        if provider in {"none", "extractive"}:
            return None
        raise ValueError(f"Unsupported generation provider: {provider}")

    def _ingest_with_parser_chain(self, path: Path):
        failures: list[str] = []
        parser_chain = getattr(
            self.settings,
            "parser_chain",
            (getattr(self.settings, "extraction_engine", "docling"),),
        )
        for engine in parser_chain:
            try:
                if engine == "docling" and path.suffix.lower() == ".pdf":
                    documents = DoclingPaperIngestor(self.settings.data_dir / "artifacts").ingest(
                        path
                    )
                elif engine == "markitdown":
                    documents = MarkItDownIngestor().ingest(path)
                elif engine == "pymupdf" and path.suffix.lower() == ".pdf":
                    documents = ingest_path(path)
                else:
                    continue
                if PARSER_TOTAL is not None:
                    PARSER_TOTAL.labels(parser=engine, outcome="success").inc()
                if failures:
                    documents = [
                        type(document)(
                            **{
                                **document.to_dict(),
                                "metadata": {
                                    **document.metadata,
                                    "parser_fallbacks": failures,
                                },
                            }
                        )
                        for document in documents
                    ]
                return documents
            except (ImportError, MemoryError, NotImplementedError, OSError, RuntimeError) as exc:
                failures.append(f"{engine}:{type(exc).__name__}")
                if PARSER_TOTAL is not None:
                    PARSER_TOTAL.labels(parser=engine, outcome="fallback").inc()
        raise RuntimeError(f"Every parser failed for {path.name}: {', '.join(failures)}")

    def ingest(self, paths: list[Path]) -> dict[str, int]:
        raw = []
        with observe(INGEST_SECONDS):
            for path in paths:
                if path.suffix.lower() in {".pdf", ".docx", ".pptx"}:
                    raw.extend(self._ingest_with_parser_chain(path))
                else:
                    raw.extend(ingest_path(path))
        chunks = chunk_documents(raw)
        existing_ids = {document.id for document in self.index.documents}
        new_lexical_chunks = [document for document in chunks if document.id not in existing_ids]
        self.index.add(new_lexical_chunks)
        self.index.save(self.index_path)
        if self.semantic is not None:
            semantic_ids = {document.id for document in self.semantic.documents}
            new_semantic_chunks = [
                document for document in chunks if document.id not in semantic_ids
            ]
            self.semantic.add(new_semantic_chunks)
            self.semantic.save(self.index_path.parent / "faiss")
        if INDEX_CHUNKS is not None:
            INDEX_CHUNKS.set(len(self.index.documents))
        return {
            "documents": len(
                {document.metadata.get("paper_id", document.source) for document in raw}
            ),
            "chunks": len(new_lexical_chunks),
            "figures": sum(document.modality == "figure" for document in new_lexical_chunks),
            "tables": sum(
                document.metadata.get("kind") == "table" for document in new_lexical_chunks
            ),
            "dataset_rows": sum(
                document.metadata.get("kind") == "supplementary_dataset"
                for document in new_lexical_chunks
            ),
            "indexed_total": len(self.index.documents),
        }

    def ask(self, question: str, top_k: int = 5):
        if self.production:
            outcome = "error"
            try:
                with observe(QUERY_SECONDS):
                    state = self.graph.invoke(question, top_k)
                outcome = "grounded" if state.get("grounded") else "abstained"
                return graph_state_to_answer(question, state)
            finally:
                if QUERY_TOTAL is not None:
                    QUERY_TOTAL.labels(
                        provider=getattr(self.settings, "generation_provider", "extractive"),
                        outcome=outcome,
                    ).inc()
        return BioRAGGraph(self.index).ask(question, top_k)

    def search(self, query: str, top_k: int = 5) -> list:
        return self.retriever.search(query, top_k)

    def metrics(self) -> dict[str, int | str]:
        documents = self.index.documents
        return {
            "backend": (
                f"faiss+bm25+{getattr(self.semantic.embeddings, 'name', type(self.semantic.embeddings).__name__)}"
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
                document.metadata.get("kind") == "supplementary_dataset" for document in documents
            ),
            "chunks": len(documents),
        }
