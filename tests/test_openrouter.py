import json
from unittest.mock import patch

import pytest

from biorag.openrouter import OpenRouterError, OpenRouterGenerator


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "Supported answer [1]."}}]}
        ).encode()


def test_openrouter_sends_grounded_request_and_strips_paper_citations(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("urllib.request.urlopen", return_value=FakeResponse()) as request:
        answer = OpenRouterGenerator(model="openrouter/auto").answer(
            "What changed?", ["Evidence text [49]."]
        )

    sent = request.call_args.args[0]
    payload = json.loads(sent.data)
    assert answer == "Supported answer [1]."
    assert payload["model"] == "openrouter/auto"
    assert "[1] Evidence text ." in payload["messages"][0]["content"]
    assert "[49]" not in payload["messages"][0]["content"]


def test_openrouter_accepts_block_content(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = FakeResponse()
    response.read = lambda: json.dumps(
        {"choices": [{"message": {"content": [{"type": "text", "text": "Answer [1]."}]}}]}
    ).encode()
    with patch("urllib.request.urlopen", return_value=response):
        answer = OpenRouterGenerator().answer("Question?", ["Evidence."])
    assert answer == "Answer [1]."


def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterGenerator()


def test_openrouter_surfaces_provider_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = FakeResponse()
    response.read = lambda: json.dumps(
        {"error": {"code": 429, "message": "rate limited"}}
    ).encode()
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(
        OpenRouterError, match="rate limited"
    ):
        OpenRouterGenerator().answer("Question?", ["Evidence."])
