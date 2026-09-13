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
    assert answer == "Supported result. [1]"
    payload = json.loads(request.call_args.args[0].data)
    assert payload["model"] == "qwen2:0.5b-instruct"
    assert payload["stream"] is False
    assert "[1] The value increased." in payload["messages"][0]["content"]


def test_ollama_embeddings_use_a_dedicated_embedding_model():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]}),
    ) as request:
        vectors = OllamaEmbeddings().embed_documents(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    payload = json.loads(request.call_args.args[0].data)
    assert payload == {"model": "nomic-embed-text", "input": ["a", "b"]}
