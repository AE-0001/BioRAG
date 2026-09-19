import json
from unittest.mock import patch

from biorag.openrouter import OpenRouterGenerator


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
