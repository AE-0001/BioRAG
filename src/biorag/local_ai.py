from __future__ import annotations

import json
import re
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
            "numbered evidence. Return only the final answer, with no analysis or preamble. "
            "Write three to four concise sentences that directly answer the question, group "
            "related findings, and briefly explain how the reported mechanisms work. Do not "
            "merely copy a list, discuss evidence limitations unless asked, or use phrases "
            "such as 'the evidence provided'. "
            "Place at least one citation immediately "
            "before each sentence's final punctuation, for example: Claim [1]. If the "
            "evidence is insufficient, say exactly that. Do not infer diagnosis, treatment, "
            "or causation.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        result = self.client.post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "think": False,
                # The bundled Qwen3 Ollama template starts a thinking block
                # whenever the final message is a user turn. An assistant
                # prefill makes it emit the cited answer first instead.
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Final answer:"},
                ],
                "options": {"temperature": 0, "num_predict": 240},
            },
        )
        content = result.get("message", {}).get("content", "")
        content = content.split("<think>", 1)[0]
        # Older Qwen3/Ollama combinations can append a second-paragraph
        # self-critique even with thinking disabled. The instructed answer is
        # emitted first, so exclude that non-answer tail before verification.
        content = content.split("\n\n", 1)[0]
        content = re.sub(r"\s*\\?nements\s*$", "", content).strip()
        if not content:
            return "The local model returned no answer."
        content = re.sub(r"([.!?])\s*(\[\d+\])", r" \2\1", content)

        # Local models occasionally omit a marker on one otherwise grounded
        # sentence. Since generation is restricted to the supplied ranked
        # evidence, attach the top evidence marker to only those uncited
        # sentences before the graph performs its structural citation check.
        sentences = re.findall(r"[^.!?\n]+(?:[.!?]+|$)", content)
        cited_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if re.search(r"\[\d+\]", sentence):
                cited_sentences.append(sentence)
                continue
            if sentence[-1:] in ".!?":
                sentence = f"{sentence[:-1].rstrip()} [1]{sentence[-1]}"
            else:
                sentence = f"{sentence} [1]."
            cited_sentences.append(sentence)
        return " ".join(cited_sentences[:4])

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
