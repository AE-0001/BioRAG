from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .gemini import GeminiGenerator
from .models import SearchHit


class ResearchState(TypedDict, total=False):
    question: str
    top_k: int
    hits: list[SearchHit]
    answer: str
    citations: list[str]
    grounded: bool
    trace: list[dict[str, str]]
    warnings: list[str]


class FourAgentResearchGraph:
    """Compiled four-agent workflow with typed, inspectable shared state."""

    def __init__(self, retriever, generator: GeminiGenerator | None = None):
        self.retriever = retriever
        self.generator = generator
        workflow = StateGraph(ResearchState)
        workflow.add_node("retrieval_agent", self._retrieval_agent)
        workflow.add_node("vision_agent", self._vision_agent)
        workflow.add_node("research_summary_agent", self._research_summary_agent)
        workflow.add_node("citation_agent", self._citation_agent)
        workflow.add_edge(START, "retrieval_agent")
        workflow.add_edge("retrieval_agent", "vision_agent")
        workflow.add_edge("vision_agent", "research_summary_agent")
        workflow.add_edge("research_summary_agent", "citation_agent")
        workflow.add_edge("citation_agent", END)
        self.compiled = workflow.compile()

    def _retrieval_agent(self, state: ResearchState) -> ResearchState:
        hits = self.retriever.search(state["question"], top_k=state.get("top_k", 5))
        return {
            "hits": hits,
            "trace": [
                *state.get("trace", []),
                {"agent": "retrieval", "action": f"retrieved {len(hits)} candidates"},
            ],
        }

    def _vision_agent(self, state: ResearchState) -> ResearchState:
        figures = [hit for hit in state["hits"] if hit.document.modality == "figure"]
        text = [hit for hit in state["hits"] if hit.document.modality == "text"]
        inspected: list[SearchHit] = []
        for hit in figures[:2]:
            image_path = hit.document.metadata.get("image_path")
            if self.generator and image_path and Path(image_path).exists():
                visual_analysis = self.generator.inspect_figure(
                    Path(image_path), hit.document.text, state["question"]
                )
                document = replace(
                    hit.document,
                    text=f"{hit.document.text}\nVisual analysis: {visual_analysis}",
                    metadata={**hit.document.metadata, "vision_analyzed": True},
                )
                hit = replace(hit, document=document)
            inspected.append(hit)
        ordered = [*inspected[:1], *text, *inspected[1:], *figures[2:]][
            : state.get("top_k", 5)
        ]
        return {
            "hits": ordered,
            "trace": [
                *state.get("trace", []),
                {"agent": "vision", "action": f"aligned {len(figures)} figure candidates"},
            ],
        }

    def _research_summary_agent(self, state: ResearchState) -> ResearchState:
        if not state["hits"]:
            answer = "I could not find sufficient evidence in the indexed literature."
            warnings = [*state.get("warnings", []), "insufficient_evidence"]
        elif self.generator:
            answer = self.generator.answer(
                state["question"], [hit.document.text for hit in state["hits"]]
            )
            warnings = state.get("warnings", [])
        else:
            answer = " ".join(
                f"{hit.document.text.split('.')[0].strip()}. [{number}]"
                for number, hit in enumerate(state["hits"][:3], start=1)
            )
            warnings = state.get("warnings", [])
        return {
            "answer": answer,
            "warnings": warnings,
            "trace": [
                *state.get("trace", []),
                {"agent": "research_summary", "action": "synthesized evidence-bound claims"},
            ],
        }

    def _citation_agent(self, state: ResearchState) -> ResearchState:
        referenced = {int(value) for value in re.findall(r"\[(\d+)\]", state["answer"])}
        valid = referenced and all(1 <= number <= len(state["hits"]) for number in referenced)
        citations = [hit.citation() for hit in state["hits"]]
        warnings = list(state.get("warnings", []))
        if not valid:
            warnings.append("citation_verification_failed")
        return {
            "citations": citations,
            "grounded": bool(valid),
            "warnings": warnings,
            "trace": [
                *state.get("trace", []),
                {"agent": "citation", "action": f"verified={bool(valid)}"},
            ],
        }

    def invoke(self, question: str, top_k: int = 5) -> ResearchState:
        return self.compiled.invoke(
            {"question": question, "top_k": top_k, "trace": [], "warnings": []}
        )
