from biorag.hybrid import ReciprocalRankFusion
from biorag.models import Document, SearchHit


def scored(document, score):
    return SearchHit(document, score, score, score)


class RankedRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.requested = None

    def search(self, query, top_k):
        self.requested = (query, top_k)
        return self.hits[:top_k]


def test_document_ranked_by_both_retrievers_wins():
    shared = Document("shared", "Shared", "Evidence", "shared.pdf")
    lexical_only = Document("lex", "Lexical", "Evidence", "lex.pdf")
    semantic_only = Document("sem", "Semantic", "Evidence", "sem.pdf")
    lexical = RankedRetriever([scored(shared, 0.8), scored(lexical_only, 1.0)])
    semantic = RankedRetriever([scored(shared, 0.7), scored(semantic_only, 1.0)])
    hits = ReciprocalRankFusion(lexical, semantic).search("query", top_k=3)
    assert hits[0].document.id == "shared"


def test_fusion_preserves_component_scores():
    document = Document("1", "Study", "Evidence", "paper.pdf")
    lexical = RankedRetriever([scored(document, 0.8)])
    semantic = RankedRetriever([scored(document, 0.6)])
    hit = ReciprocalRankFusion(lexical, semantic).search("query", 1)[0]
    assert hit.lexical_score == 0.8
    assert hit.semantic_score == 0.6


def test_fusion_respects_top_k():
    docs = [Document(str(i), "Study", "Evidence", f"{i}.pdf") for i in range(4)]
    fusion = ReciprocalRankFusion(
        RankedRetriever([scored(doc, 1.0) for doc in docs]), RankedRetriever([])
    )
    assert len(fusion.search("query", top_k=2)) == 2


def test_candidate_pool_is_larger_than_requested_results():
    lexical = RankedRetriever([])
    semantic = RankedRetriever([])
    ReciprocalRankFusion(lexical, semantic).search("marker", top_k=2)
    assert lexical.requested == ("marker", 20)
    assert semantic.requested == ("marker", 20)


def test_empty_rankings_return_empty_result():
    assert ReciprocalRankFusion(RankedRetriever([]), RankedRetriever([])).search("q") == []

