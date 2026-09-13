from __future__ import annotations

import math
import mimetypes
import os
from collections.abc import Iterable
from pathlib import Path

from .models import Document


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class GeminiEmbeddings:
    """LangChain-compatible Gemini retrieval embeddings."""

    def __init__(self, model: str | None = None, dimensions: int = 768):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai to use Gemini embeddings") from exc
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before using Gemini embeddings")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("BIORAG_EMBEDDING_MODEL", "gemini-embedding-2")
        self.name = f"gemini:{self.model}"
        self.dimensions = dimensions

    def _embed(self, texts: Iterable[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        values = list(texts)
        if self.model == "gemini-embedding-2":
            if task_type == "RETRIEVAL_QUERY":
                values = [f"task: search result | query: {value}" for value in values]
            else:
                values = [f"title: none | text: {value}" for value in values]
            # Embedding 2 aggregates a plain list into one vector. Explicit
            # Content objects request one vector per independently indexed item.
            contents = [types.Content(parts=[types.Part.from_text(text=value)]) for value in values]
            config = types.EmbedContentConfig(output_dimensionality=self.dimensions)
        else:
            contents = values
            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensions,
            )
        response = self.client.models.embed_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return [_normalize(list(item.values)) for item in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def embed_evidence(self, documents: list[Document]) -> list[list[float]]:
        """Embed text or image+caption into one Gemini multimodal space."""
        from google.genai import types

        output: list[list[float]] = []
        pending_text: list[str] = []

        def flush_text_batch() -> None:
            """Use a few batched API calls, not one request for every chunk."""
            while pending_text:
                batch = pending_text[:50]
                del pending_text[:50]
                output.extend(self._embed(batch, "RETRIEVAL_DOCUMENT"))

        for document in documents:
            image_path = document.metadata.get("image_path")
            image_embeddings_enabled = os.getenv("BIORAG_IMAGE_EMBEDDINGS", "false").lower() in {
                "1",
                "true",
                "yes",
            }
            if (
                image_embeddings_enabled
                and document.modality == "figure"
                and image_path
                and Path(image_path).exists()
            ):
                flush_text_batch()
                mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
                contents = [
                    f"{document.title}\n{document.text}",
                    types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type=mime_type),
                ]
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
                )
                output.append(_normalize(list(response.embeddings[0].values)))
            else:
                pending_text.append(f"{document.title}\n{document.text}")
        flush_text_batch()
        return output


class GeminiGenerator:
    """Grounded answer synthesis with strict evidence-only prompting."""

    supports_vision = True

    def __init__(self, model: str | None = None):
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before using Gemini generation")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("BIORAG_GENERATION_MODEL", "gemini-3.5-flash")

    def answer(self, question: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(
            f"[{number}] {context}" for number, context in enumerate(contexts, start=1)
        )
        prompt = f"""You are a biomedical research assistant, not a clinician.
Use only the numbered evidence below. Cite every factual claim with [n].
If evidence is insufficient, say so. Distinguish association from causation.

Question: {question}

Evidence:
{evidence}
"""
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text or "The model returned no answer."

    def inspect_figure(self, image_path: Path, caption: str, question: str) -> str:
        from google.genai import types

        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        prompt = f"""Inspect this biomedical research figure in relation to the question.
Report only directly visible trends, labels, axes, values, and uncertainty.
Do not infer causation or clinical advice.

Question: {question}
Existing caption: {caption}
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type),
            ],
        )
        return response.text or caption
