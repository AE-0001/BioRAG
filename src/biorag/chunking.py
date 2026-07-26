from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Document

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def semantic_chunks(text: str, max_chars: int = 900, overlap_sentences: int = 1) -> list[str]:
    """Split on sentence boundaries while retaining a small evidence overlap."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(normalized) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        proposed = " ".join([*current, sentence])
        if current and len(proposed) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(documents: Iterable[Document], max_chars: int = 900) -> list[Document]:
    output: list[Document] = []
    for document in documents:
        parts = semantic_chunks(document.text, max_chars=max_chars)
        for index, part in enumerate(parts):
            output.append(
                Document(
                    id=f"{document.id}:chunk:{index}",
                    title=document.title,
                    text=part,
                    source=document.source,
                    modality=document.modality,
                    page=document.page,
                    metadata={**document.metadata, "parent_id": document.id, "chunk": index},
                )
            )
    return output

