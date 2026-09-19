from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .models import SearchHit
from .observability import RETRIEVAL_ATTEMPTS, RETRIEVAL_SECONDS, observe


class ResearchState(TypedDict, total=False):
    question: str
    retrieval_query: str
    top_k: int
    hits: list[SearchHit]
    answer: str
    citations: list[str]
    grounded: bool
    trace: list[dict[str, str]]
    warnings: list[str]
    retrieval_attempts: int
    citation_attempts: int
    evidence_sufficient: bool
    route: str


class FourAgentResearchGraph:
    """Conditional RAG graph with retrieval correction, vision routing, and abstention."""

    def __init__(
        self,
        retriever,
        generator=None,
        minimum_relevance: float = 0.20,
        max_retrieval_attempts: int = 2,
    ):
        self.retriever = retriever
        self.generator = generator
        self.minimum_relevance = minimum_relevance
        self.max_retrieval_attempts = max(1, max_retrieval_attempts)
        workflow = StateGraph(ResearchState)
        workflow.add_node("retrieval_agent", self._retrieval_agent)
        workflow.add_node("evidence_grader", self._evidence_grader)
        workflow.add_node("query_rewriter", self._query_rewriter)
        workflow.add_node("vision_agent", self._vision_agent)
        workflow.add_node("research_summary_agent", self._research_summary_agent)
        workflow.add_node("citation_agent", self._citation_agent)
        workflow.add_node("abstain", self._abstain)
        workflow.add_edge(START, "retrieval_agent")
        workflow.add_edge("retrieval_agent", "evidence_grader")
        workflow.add_conditional_edges(
            "evidence_grader",
            self._route_after_grading,
            {
                "rewrite": "query_rewriter",
                "vision": "vision_agent",
                "answer": "research_summary_agent",
                "abstain": "abstain",
            },
        )
        workflow.add_edge("query_rewriter", "retrieval_agent")
        workflow.add_edge("vision_agent", "research_summary_agent")
        workflow.add_edge("research_summary_agent", "citation_agent")
        workflow.add_conditional_edges(
            "citation_agent",
            self._route_after_citation,
            {"retry": "research_summary_agent", "done": END, "abstain": "abstain"},
        )
        workflow.add_edge("abstain", END)
        self.compiled = workflow.compile()

    def _retrieval_agent(self, state: ResearchState) -> ResearchState:
        query = state.get("retrieval_query", state["question"])
        with observe(RETRIEVAL_SECONDS):
            hits = self.retriever.search(query, top_k=state.get("top_k", 5))
        attempts = state.get("retrieval_attempts", 0) + 1
        return {
            "hits": hits,
            "retrieval_attempts": attempts,
            "trace": [
                *state.get("trace", []),
                {"agent": "retrieval", "action": f"attempt={attempts}; candidates={len(hits)}"},
            ],
        }

    def _evidence_grader(self, state: ResearchState) -> ResearchState:
        best = max(
            (
                max(hit.score, hit.lexical_score, hit.semantic_score)
                for hit in state.get("hits", [])
            ),
            default=0.0,
        )
        sufficient = bool(state.get("hits")) and best >= self.minimum_relevance
        return {
            "evidence_sufficient": sufficient,
            "trace": [
                *state.get("trace", []),
                {
                    "agent": "evidence_grader",
                    "action": f"best_score={best:.3f}; sufficient={sufficient}",
                },
            ],
        }

    def _route_after_grading(
        self, state: ResearchState
    ) -> Literal["rewrite", "vision", "answer", "abstain"]:
        if not state.get("evidence_sufficient"):
            return (
                "rewrite"
                if state.get("retrieval_attempts", 0) < self.max_retrieval_attempts
                else "abstain"
            )
        wants_visual = any(
            term in state["question"].lower()
            for term in ("figure", "image", "plot", "chart", "panel")
        )
        has_figure = any(hit.document.modality == "figure" for hit in state.get("hits", []))
        return "vision" if wants_visual and has_figure else "answer"

    def _query_rewriter(self, state: ResearchState) -> ResearchState:
        return {
            "retrieval_query": f"{state['question']} biomedical study evidence findings limitations",
            "trace": [
                *state.get("trace", []),
                {"agent": "query_rewriter", "action": "expanded biomedical retrieval query"},
            ],
        }

    def _vision_agent(self, state: ResearchState) -> ResearchState:
        figures = [hit for hit in state["hits"] if hit.document.modality == "figure"]
        text = [hit for hit in state["hits"] if hit.document.modality == "text"]
        inspected: list[SearchHit] = []
        supports_vision = bool(
            self.generator
            and hasattr(self.generator, "inspect_figure")
            and getattr(self.generator, "supports_vision", True)
        )
        for hit in figures[:2]:
            image_path = hit.document.metadata.get("image_path")
            if supports_vision and image_path and Path(image_path).exists():
                analysis = self.generator.inspect_figure(
                    Path(image_path), hit.document.text, state["question"]
                )
                hit = replace(
                    hit,
                    document=replace(
                        hit.document,
                        text=f"{hit.document.text}\nVisual analysis: {analysis}",
                        metadata={**hit.document.metadata, "vision_analyzed": True},
                    ),
                )
            inspected.append(hit)
        ordered = [*inspected[:1], *text, *inspected[1:], *figures[2:]][: state.get("top_k", 5)]
        return {
            "hits": ordered,
            "trace": [
                *state.get("trace", []),
                {
                    "agent": "vision",
                    "action": f"figures={len(figures)}; inspected={supports_vision}",
                },
            ],
        }

    def _research_summary_agent(self, state: ResearchState) -> ResearchState:
        if self.generator:
            answer = self.generator.answer(
                state["question"], [hit.document.text for hit in state["hits"]]
            )
        else:
            answer = " ".join(
                f"{hit.document.text.split('.')[0].strip()}. [{number}]"
                for number, hit in enumerate(state["hits"][:3], start=1)
            )
        return {
            "answer": answer,
            "trace": [
                *state.get("trace", []),
                {"agent": "research_summary", "action": "synthesized evidence-bound claims"},
            ],
        }

    def _citation_agent(self, state: ResearchState) -> ResearchState:
        referenced = {int(value) for value in re.findall(r"\[(\d+)\]", state["answer"])}
        indices_valid = bool(referenced) and all(1 <= n <= len(state["hits"]) for n in referenced)
        # Models place citations either before punctuation ("claim [1].") or
        # immediately after it ("claim. [1]"). Treat both as the same claim
        # without accepting a single citation as coverage for an entire answer.
        factual = [
            claim.strip()
            for claim in re.findall(
                r"[^.!?\n]+(?:[.!?]+|$)(?:\s*\[\d+\])?",
                state["answer"],
            )
            if claim.strip() and "insufficient" not in claim.lower()
        ]
        claims_cited = bool(factual) and all(
            re.search(r"\[\d+\]", sentence) for sentence in factual
        )
        valid = indices_valid and claims_cited
        attempts = state.get("citation_attempts", 0) + 1
        warnings = [w for w in state.get("warnings", []) if w != "citation_verification_failed"]
        if not valid:
            warnings.append("citation_verification_failed")
        citations = [
            state["hits"][n - 1].citation()
            for n in sorted(referenced)
            if 1 <= n <= len(state["hits"])
        ]
        return {
            "citations": citations,
            "grounded": valid,
            "citation_attempts": attempts,
            "warnings": warnings,
            "trace": [
                *state.get("trace", []),
                {"agent": "citation", "action": f"claim_coverage_verified={valid}"},
            ],
        }

    def _route_after_citation(self, state: ResearchState) -> Literal["retry", "done", "abstain"]:
        if state.get("grounded"):
            return "done"
        # Relevant evidence exists and the answer is still useful, so preserve
        # it with grounded=False rather than replacing it with an abstention.
        if state.get("evidence_sufficient") and state.get("answer"):
            return "done"
        return "retry" if self.generator and state.get("citation_attempts", 0) < 2 else "abstain"

    def _abstain(self, state: ResearchState) -> ResearchState:
        warnings = list(state.get("warnings", []))
        if not state.get("evidence_sufficient"):
            warnings.append("insufficient_evidence")
            warnings.append("citation_verification_failed")
        if RETRIEVAL_ATTEMPTS is not None:
            RETRIEVAL_ATTEMPTS.observe(state.get("retrieval_attempts", 0))
        return {
            "answer": "I could not find sufficient, citation-verifiable evidence in the indexed literature.",
            "citations": [],
            "grounded": False,
            "warnings": list(dict.fromkeys(warnings)),
            "route": "abstain",
            "trace": [
                *state.get("trace", []),
                {"agent": "abstain", "action": "refused unsupported answer"},
            ],
        }

    def invoke(self, question: str, top_k: int = 5) -> ResearchState:
        result = self.compiled.invoke(
            {
                "question": question,
                "retrieval_query": question,
                "top_k": top_k,
                "trace": [],
                "warnings": [],
                "retrieval_attempts": 0,
                "citation_attempts": 0,
            }
        )
        if RETRIEVAL_ATTEMPTS is not None and result.get("route") != "abstain":
            RETRIEVAL_ATTEMPTS.observe(result.get("retrieval_attempts", 0))
        return result
