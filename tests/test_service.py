from pathlib import Path
from types import SimpleNamespace

from biorag.models import Document, SearchHit
import biorag.service as service_module
from biorag.service import BioRAGService, graph_state_to_answer


class RecordingSemanticStore:
    def __init__(self) -> None:
        self.documents = []

    def add(self, documents) -> None:
        self.documents.extend(documents)

    def save(self, directory: Path) -> None:
        return None


def test_ingest_backfills_semantic_index_for_existing_lexical_chunks(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text("B7-H3 was associated with response.", encoding="utf-8")
    service = BioRAGService(tmp_path / "index.json")

    service.ingest([source])
    semantic = RecordingSemanticStore()
    service.semantic = semantic
    service.ingest([source])

    assert len(service.index.documents) == 1
    assert len(semantic.documents) == 1


def test_graph_state_is_normalized_to_interface_answer_contract():
    hit = SearchHit(Document("one", "Study", "Evidence", "paper.pdf"), 0.9, 0.2, 0.8)
    answer = graph_state_to_answer(
        "Question?",
        {
            "answer": "Evidence-backed answer. [1]",
            "citations": ["Study (paper.pdf)"],
            "hits": [hit],
            "grounded": True,
            "trace": [{"agent": "retrieval", "action": "retrieved evidence"}],
        },
    )

    assert answer.question == "Question?"
    assert answer.evidence == [hit]
    assert answer.trace[0].agent == "retrieval"


def test_gemini_failure_falls_back_to_local_graph(monkeypatch):
    class LocalGraph:
        def invoke(self, question, top_k, history=None):
            return {
                "answer": "Local answer [1].",
                "citations": [],
                "hits": [],
                "grounded": True,
                "trace": [],
            }

    class FailingCloudGraph:
        def invoke(self, question, top_k, history=None):
            raise ConnectionError("cloud unavailable")

    service = BioRAGService.__new__(BioRAGService)
    service.production = True
    service.settings = SimpleNamespace(
        generation_provider="ollama", minimum_relevance=0.2, max_retrieval_attempts=2
    )
    service.generator = object()
    service.retriever = object()
    service.graph = LocalGraph()
    monkeypatch.setattr(service, "_generator_for_provider", lambda provider: object())
    monkeypatch.setattr(
        service_module, "FourAgentResearchGraph", lambda *args, **kwargs: FailingCloudGraph()
    )

    answer = service.ask("Question?", generation_provider="gemini")

    assert answer.answer == "Local answer [1]."
    assert answer.trace[0].agent == "provider_router"
    assert "fell back to ollama" in answer.trace[0].action
