from biorag.models import Document
from biorag.retrieval import HybridIndex


def test_hybrid_search_ranks_relevant_evidence_first():
    index = HybridIndex()
    index.add(
        [
            Document("1", "Oncology", "B7-H3 expression predicts response.", "paper-a"),
            Document("2", "Cardiology", "Blood pressure was measured.", "paper-b"),
        ]
    )
    hits = index.search("Does B7-H3 predict treatment response?", top_k=2)
    assert hits[0].document.id == "1"
    assert hits[0].score > hits[1].score


def test_index_round_trip(tmp_path):
    index = HybridIndex()
    index.add([Document("1", "Title", "Evidence", "source")])
    path = tmp_path / "index.json"
    index.save(path)
    restored = HybridIndex.load(path)
    assert restored.documents == index.documents

