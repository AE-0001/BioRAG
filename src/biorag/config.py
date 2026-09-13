from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Makes the documented `copy .env.example .env` setup work for local runs,
# while preserving environment variables supplied by Docker or CI.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("BIORAG_DATA_DIR", "data"))
    index_dir: Path = Path(os.getenv("BIORAG_INDEX_DIR", "data/index"))
    embedding_provider: str = os.getenv("BIORAG_EMBEDDING_PROVIDER", "ollama")
    embedding_model: str = os.getenv("BIORAG_EMBEDDING_MODEL", "gemini-embedding-2")
    generation_provider: str = os.getenv(
        "BIORAG_GENERATION_PROVIDER",
        "gemini" if os.getenv("BIORAG_GENERATION_MODEL", "").startswith("gemini") else "ollama",
    )
    generation_model: str = os.getenv("BIORAG_GENERATION_MODEL", "qwen2:0.5b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_embedding_model: str = os.getenv("BIORAG_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    request_timeout: float = float(os.getenv("BIORAG_REQUEST_TIMEOUT", "60"))
    # A comma-separated, ordered parser chain. Docling is the primary scientific
    # parser; MarkItDown and PyMuPDF are explicit degradation paths.
    extraction_engines: str = os.getenv("BIORAG_EXTRACTION_ENGINES", "docling,markitdown,pymupdf")
    top_k: int = int(os.getenv("BIORAG_TOP_K", "5"))
    minimum_relevance: float = float(os.getenv("BIORAG_MIN_RELEVANCE", "0.20"))
    max_retrieval_attempts: int = int(os.getenv("BIORAG_MAX_RETRIEVAL_ATTEMPTS", "2"))

    @property
    def parser_chain(self) -> tuple[str, ...]:
        return tuple(
            engine.strip().lower()
            for engine in self.extraction_engines.split(",")
            if engine.strip()
        )

    @property
    def extraction_engine(self) -> str:
        """Backwards-compatible name for older callers and configuration tests."""
        return self.parser_chain[0] if self.parser_chain else "docling"

    @property
    def has_gemini_key(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
