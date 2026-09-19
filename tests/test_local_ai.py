import json
from unittest.mock import patch

from biorag.local_ai import OllamaEmbeddings, OllamaGenerator


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_ollama_qwen_generator_sends_grounded_non_streaming_chat():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse({"message": {"content": "Supported result. [1]"}}),
    ) as request:
        answer = OllamaGenerator().answer("What changed?", ["The value increased."])
    assert answer == "Supported result [1]."
    payload = json.loads(request.call_args.args[0].data)
    assert payload["model"] == "qwen3:4b"
    assert payload["stream"] is False
    assert "[1] The value increased." in payload["messages"][0]["content"]
    assert payload["messages"][-1] == {
        "role": "assistant",
        "content": "Final answer:",
    }


def test_ollama_qwen_generator_removes_trailing_thinking_content():
    payload = {
        "message": {
            "content": "Supported result [1].\n\nWait, I should reconsider.\n<think>analysis"
        }
    }
    with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
        answer = OllamaGenerator().answer("What changed?", ["The value increased."])
    assert answer == "Supported result [1]."


def test_ollama_qwen_generator_preserves_uncited_text_for_graph_review():
    payload = {
        "message": {
            "content": "First supported claim. Second supported claim [2]."
        }
    }
    with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
        answer = OllamaGenerator().answer("What changed?", ["Supporting evidence."])
    assert answer == "First supported claim. Second supported claim [2]."


def test_ollama_qwen_generator_removes_source_bibliography_markers():
    payload = {"message": {"content": "Glycopeptides inhibit synthesis [1]."}}
    contexts = [
        "They bind peptidoglycan precursors [49] and block synthesis [72-74]."
    ]
    with patch("urllib.request.urlopen", return_value=FakeResponse(payload)) as request:
        answer = OllamaGenerator().answer("How do glycopeptides work?", contexts)

    prompt = json.loads(request.call_args.args[0].data)["messages"][0]["content"]
    assert answer == "Glycopeptides inhibit synthesis [1]."
    assert "[1] They bind peptidoglycan precursors" in prompt
    assert "[49]" not in prompt
    assert "[72-74]" not in prompt


def test_ollama_embeddings_use_a_dedicated_embedding_model():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]}),
    ) as request:
        vectors = OllamaEmbeddings().embed_documents(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    payload = json.loads(request.call_args.args[0].data)
    assert payload == {"model": "nomic-embed-text", "input": ["a", "b"]}
