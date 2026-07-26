from __future__ import annotations

import re

from .models import AgentTrace, Answer, SearchHit
from .retrieval import HybridIndex, tokenize

SENTENCES = re.compile(r"(?<=[.!?])\s+")
INTENT_TERMS = {
    "limitation": {"small", "randomized", "bias", "confound", "validation", "limitation"},
    "limitations": {"small", "randomized", "bias", "confound", "validation", "limitation"},
    "rate": {"rate", "percent", "percentage"},
    "figure": {"figure", "chart", "plot", "image"},
}


class RetrievalAgent:
    def __init__(self, index: HybridIndex):
        self.index = index

    def run(self, question: str, top_k: int) -> list[SearchHit]:
        return self.index.search(question, top_k=top_k)


class VisionAgent:
    def run(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Promote a relevant visual hit without letting it swamp textual evidence."""
        figures = [hit for hit in hits if hit.document.modality == "figure"]
        if not figures or figures[0].score <= 0:
            return hits
        best_figure = figures[0]
        return [best_figure, *[hit for hit in hits if hit is not best_figure]]


class SummaryAgent:
    def run(self, question: str, hits: list[SearchHit]) -> str:
        if not hits or hits[0].score <= 0:
            return "I could not find sufficient evidence in the indexed literature."
        query_terms = set(tokenize(question))
        for term in list(query_terms):
            query_terms.update(INTENT_TERMS.get(term, set()))
        candidates: list[tuple[int, int, str]] = []
        for number, hit in enumerate(hits[:3], start=1):
            sentences = [part.strip() for part in SENTENCES.split(hit.document.text) if part.strip()]
            for sentence in sentences:
                overlap = len(query_terms.intersection(tokenize(sentence)))
                if overlap:
                    candidates.append((overlap, number, sentence))
        selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:3]
        if not selected:
            return "I could not find sufficient evidence in the indexed literature."
        return " ".join(f"{sentence} [{number}]" for _, number, sentence in selected)


class CitationAgent:
    def run(self, answer: str, hits: list[SearchHit]) -> tuple[list[str], bool]:
        markers = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
        citations = [hit.citation() for hit in hits]
        markers_present = bool(markers) and all(1 <= marker <= len(hits) for marker in markers)
        evidence_relevant = bool(hits and hits[0].score > 0)
        return citations, markers_present and evidence_relevant


class BioRAGGraph:
    """Explicit agent orchestration whose trace is inspectable in demos and tests."""

    def __init__(self, index: HybridIndex):
        self.retrieval = RetrievalAgent(index)
        self.vision = VisionAgent()
        self.summary = SummaryAgent()
        self.citation = CitationAgent()

    def ask(self, question: str, top_k: int = 5) -> Answer:
        trace: list[AgentTrace] = []
        hits = self.retrieval.run(question, top_k)
        trace.append(AgentTrace("retrieval", "hybrid_search", f"returned {len(hits)} hits"))
        hits = self.vision.run(hits)
        figure_count = sum(hit.document.modality == "figure" for hit in hits)
        trace.append(AgentTrace("vision", "rank_visual_evidence", f"{figure_count} figure hits"))
        response = self.summary.run(question, hits)
        trace.append(AgentTrace("summary", "compose", "extractive, evidence-only answer"))
        citations, grounded = self.citation.run(response, hits)
        trace.append(AgentTrace("citation", "verify", f"grounded={grounded}"))
        return Answer(question, response, citations, hits, grounded, trace)
