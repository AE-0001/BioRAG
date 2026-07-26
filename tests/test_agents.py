from biorag.agents import BioRAGGraph
from biorag.models import Document
from biorag.retrieval import HybridIndex


def test_graph_returns_cited_grounded_answer():
    index = HybridIndex()
    index.add(
        [
            Document(
                "1",
                "BR-01",
                "High B7-H3 expression was associated with higher observed response.",
                "study.pdf",
                page=2,
            )
        ]
    )
    answer = BioRAGGraph(index).ask("What was B7-H3 associated with?", top_k=1)
    assert answer.grounded
    assert "[1]" in answer.answer
    assert "p. 2" in answer.citations[0]
    assert [step.agent for step in answer.trace] == [
        "retrieval",
        "vision",
        "summary",
        "citation",
    ]

