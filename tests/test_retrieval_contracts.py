import math

import pytest

from biorag.models import Document
from biorag.retrieval import HashEmbedding, HybridIndex, cosine, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B7-H3 response", ["b7-h3", "response"]),
        ("CD8+ T-cells", ["cd8", "t-cells"]),
        ("A", []),
        ("Mixed CASE Terms", ["mixed", "case", "terms"]),
    ],
)
def test_tokenization_contract(text, expected):
    assert tokenize(text) == expected


@pytest.mark.parametrize("weight", [-0.01, 1.01, -100, 100])
def test_semantic_weight_outside_unit_interval_is_rejected(weight):
    with pytest.raises(ValueError, match="semantic_weight"):
        HybridIndex(weight)


def test_hash_embedding_is_deterministic():
    embedder = HashEmbedding(dimensions=32)
    assert embedder.encode("same biomedical text") == embedder.encode("same biomedical text")


def test_nonempty_hash_embedding_has_unit_norm():
    vector = HashEmbedding(dimensions=32).encode("biomarker response cohort")
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_empty_hash_embedding_is_zero_vector():
    assert HashEmbedding(dimensions=4).encode("") == [0.0, 0.0, 0.0, 0.0]


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])


def test_empty_query_returns_no_results():
    index = HybridIndex()
    index.add([Document("1", "Title", "Evidence", "paper.pdf")])
    assert index.search("   ") == []


def test_empty_index_returns_no_results():
    assert HybridIndex().search("evidence") == []


def test_top_k_is_respected():
    index = HybridIndex()
    index.add([Document(str(i), f"Paper {i}", "shared evidence", f"{i}.pdf") for i in range(5)])
    assert len(index.search("shared evidence", top_k=2)) == 2


def test_exact_rare_term_has_lexical_signal():
    index = HybridIndex()
    index.add([
        Document("1", "Marker", "The cohort expressed B7-H3.", "a.pdf"),
        Document("2", "Control", "The cohort was observed.", "b.pdf"),
    ])
    hit = index.search("B7-H3", top_k=1)[0]
    assert hit.document.id == "1"
    assert hit.lexical_score == pytest.approx(1.0)


def test_saved_index_preserves_weight_and_ranking(tmp_path):
    index = HybridIndex(semantic_weight=0.25)
    index.add([
        Document("1", "Oncology", "B7-H3 response", "a.pdf"),
        Document("2", "Cardiology", "blood pressure", "b.pdf"),
    ])
    path = tmp_path / "index.json"
    index.save(path)
    restored = HybridIndex.load(path)
    assert restored.semantic_weight == 0.25
    assert restored.search("B7-H3")[0].document.id == "1"

