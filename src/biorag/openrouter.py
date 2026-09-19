from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot generate an answer."""


class OpenRouterGenerator:
    """Evidence-grounded generation through OpenRouter's compatible API."""

    supports_vision = False

    def __init__(self, model: str | None = None, timeout: float = 60):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY before using OpenRouter")
        self.model = model or os.getenv("BIORAG_OPENROUTER_MODEL", "openrouter/auto")
        self.timeout = timeout

    def answer(self, question: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(
            "[{}] {}".format(
                number, re.sub(r"\[(?:\d+[\s,–-]*)+\]", "", context)
            )
            for number, context in enumerate(contexts, start=1)
        )
        prompt = (
            "Use only the numbered biomedical evidence below. Write three to four "
            "concise sentences and cite every factual claim with the matching [n]. "
            "Do not provide clinical advice. If evidence is insufficient, say so.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 800,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AE-0001/BioRAG",
                "X-Title": "BioRAG",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        if error := payload.get("error"):
            if isinstance(error, dict):
                error = error.get("message") or error.get("code") or "provider error"
            raise OpenRouterError(f"OpenRouter provider error: {error}")

        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(
                f"OpenRouter response had no answer choices; model={payload.get('model', self.model)}"
            ) from exc

        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            if answer := "\n".join(parts).strip():
                return answer

        finish_reason = choice.get("finish_reason", "unknown")
        routed_model = payload.get("model", self.model)
        raise OpenRouterError(
            "OpenRouter returned no answer text "
            f"(model={routed_model}, finish_reason={finish_reason})"
        )
