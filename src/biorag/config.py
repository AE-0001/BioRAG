from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("BIORAG_DATA_DIR", "data"))
    index_dir: Path = Path(os.getenv("BIORAG_INDEX_DIR", "data/index"))
    embedding_model: str = os.getenv("BIORAG_EMBEDDING_MODEL", "gemini-embedding-2")
    generation_model: str = os.getenv("BIORAG_GENERATION_MODEL", "gemini-3.5-flash")
    extraction_engine: str = os.getenv("BIORAG_EXTRACTION_ENGINE", "docling")
    top_k: int = int(os.getenv("BIORAG_TOP_K", "5"))
    minimum_relevance: float = float(os.getenv("BIORAG_MIN_RELEVANCE", "0.20"))

    @property
    def has_gemini_key(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

