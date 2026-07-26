from __future__ import annotations

from .models import SearchHit
from .retrieval import HybridIndex


class ReciprocalRankFusion:
    """Fuse lexical BM25 and semantic FAISS rankings without score calibration."""

    def __init__(self, lexical: HybridIndex, semantic, rank_constant: int = 60):
        self.lexical = lexical
        self.semantic = semantic
        self.rank_constant = rank_constant

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        candidate_k = max(top_k * 4, 20)
        lexical_hits = self.lexical.search(query, candidate_k)
        semantic_hits = self.semantic.search(query, candidate_k)
        by_id: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        lexical_scores: dict[str, float] = {}
        semantic_scores: dict[str, float] = {}
        for ranking, hits in (("lexical", lexical_hits), ("semantic", semantic_hits)):
            for rank, hit in enumerate(hits, start=1):
                document_id = hit.document.id
                by_id[document_id] = hit
                scores[document_id] = scores.get(document_id, 0.0) + 1 / (
                    self.rank_constant + rank
                )
                if ranking == "lexical":
                    lexical_scores[document_id] = hit.score
                else:
                    semantic_scores[document_id] = hit.score
        fused = [
            SearchHit(
                document=hit.document,
                score=scores[document_id],
                lexical_score=lexical_scores.get(document_id, 0.0),
                semantic_score=semantic_scores.get(document_id, 0.0),
            )
            for document_id, hit in by_id.items()
        ]
        return sorted(fused, key=lambda hit: hit.score, reverse=True)[:top_k]

