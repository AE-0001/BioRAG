from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot satisfy a request."""


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, endpoint: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(
                f"Ollama request failed at {self.base_url}. Start Ollama and pull the configured model."
            ) from exc


class OllamaGenerator:
    """Evidence-only local answer generation, defaulting to the small Qwen baseline."""

    supports_vision = False

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        timeout: float = 60,
    ):
        self.model = model
        self.client = OllamaClient(base_url, timeout)

    def answer(self, question: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(
            f"[{number}] {context}" for number, context in enumerate(contexts, start=1)
        )
        prompt = (
            "You are a biomedical research assistant, not a clinician. Use only the "
            "numbered evidence. Cite every factual claim with [n]. If the evidence is "
            "insufficient, say exactly that. Do not infer diagnosis, treatment, or causation.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        result = self.client.post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0},
            },
        )
        return result.get("message", {}).get("content", "").strip() or (
            "The local model returned no answer."
        )

    def inspect_figure(self, image_path, caption: str, question: str) -> str:
        # Qwen3 4B is text-only. The graph retains captions and routes actual
        # image inspection only to a provider that declares vision capability.
        return caption


class OllamaEmbeddings:
    """Local neural embeddings through Ollama's batch embedding endpoint."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60,
        batch_size: int = 32,
    ):
        self.model = model
        self.name = f"ollama:{model}"
        self.client = OllamaClient(base_url, timeout)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            result = self.client.post("/api/embed", {"model": self.model, "input": batch})
            embeddings.extend(result["embeddings"])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_evidence(self, documents) -> list[list[float]]:
        return self.embed_documents(
            [f"{document.title}\n{document.text}" for document in documents]
        )
