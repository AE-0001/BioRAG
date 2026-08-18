from typing import ClassVar

import pytest

from biorag.faiss_store import FaissStore
from biorag.models import Document


class FakeEmbeddings:
    vectors: ClassVar[dict[str, list[float]]] = {
        "oncology": [1.0, 0.0, 0.0],
        "cardiology": [0.0, 1.0, 0.0],
    }

    def embed_documents(self, texts):
        return [self.vectors["oncology"] if "B7-H3" in text else self.vectors["cardiology"]
                for text in texts]

    def embed_query(self, text):
        return self.vectors["oncology"] if "B7-H3" in text else self.vectors["cardiology"]


class MultimodalEmbeddings(FakeEmbeddings):
    def __init__(self):
        self.received = None

    def embed_evidence(self, documents):
        self.received = documents
        return [[1.0, 0.0, 0.0] for _ in documents]


def documents():
    return [
        Document("1", "Oncology", "B7-H3 response", "a.pdf"),
        Document("2", "Cardiology", "Blood pressure", "b.pdf"),
    ]


def test_empty_store_returns_no_hits():
    assert FaissStore(FakeEmbeddings()).search("B7-H3") == []


def test_empty_add_is_noop():
    store = FaissStore(FakeEmbeddings())
    store.add([])
    assert store.index is None


def test_semantic_search_ranks_expected_document():
    store = FaissStore(FakeEmbeddings())
    store.add(documents())
    assert store.search("B7-H3", top_k=1)[0].document.id == "1"


def test_search_respects_top_k():
    store = FaissStore(FakeEmbeddings())
    store.add(documents())
    assert len(store.search("B7-H3", top_k=1)) == 1


def test_multimodal_provider_receives_documents():
    embeddings = MultimodalEmbeddings()
    store = FaissStore(embeddings)
    store.add(documents())
    assert embeddings.received == documents()


def test_dimension_change_requires_rebuild():
    embeddings = FakeEmbeddings()
    store = FaissStore(embeddings)
    store.add(documents()[:1])
    embeddings.embed_documents = lambda texts: [[1.0, 0.0] for _ in texts]
    with pytest.raises(ValueError, match="dimensionality"):
        store.add(documents()[1:])


def test_save_without_index_creates_nothing(tmp_path):
    directory = tmp_path / "faiss"
    FaissStore(FakeEmbeddings()).save(directory)
    assert not directory.exists()


def test_store_round_trip_preserves_documents_and_ranking(tmp_path):
    directory = tmp_path / "faiss"
    original = FaissStore(FakeEmbeddings())
    original.add(documents())
    original.save(directory)
    restored = FaissStore(FakeEmbeddings())
    restored.load(directory)
    assert restored.documents == documents()
    assert restored.search("B7-H3", 1)[0].document.id == "1"
