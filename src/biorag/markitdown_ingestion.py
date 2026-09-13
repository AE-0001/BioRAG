from __future__ import annotations

from pathlib import Path

from .ingestion import stable_id
from .models import Document


class MarkItDownIngestor:
    """Fast local fallback that normalizes common documents into Markdown."""

    def ingest(self, path: Path) -> list[Document]:
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise RuntimeError("Install markitdown to use this parser") from exc
        result = MarkItDown(enable_plugins=False).convert(str(path))
        text = (result.text_content or "").strip()
        if not text:
            raise RuntimeError(f"MarkItDown extracted no content from {path.name}")
        return [
            Document(
                id=stable_id(str(path.resolve()), text),
                title=path.stem.replace("_", " ").title(),
                text=text,
                source=path.name,
                metadata={"kind": "paper_text", "parser": "markitdown"},
            )
        ]
