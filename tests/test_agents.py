from biorag.agents import BioRAGGraph, SummaryAgent
from biorag.models import Document, SearchHit
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


def test_offline_summary_does_not_emit_markdown_headings():
    hit = SearchHit(
        Document(
            "heading",
            "Study",
            "# Biomarker study\nB7-H3 was associated with response.",
            "paper.md",
        ),
        score=0.9,
        lexical_score=0.9,
        semantic_score=0.0,
    )

    answer = SummaryAgent().run("What was associated with response?", [hit])

    assert "#" not in answer
    assert "B7-H3 was associated with response." in answer


def test_offline_summary_uses_multiple_evidence_sources_when_available():
    hits = [
        SearchHit(Document("one", "One", "B7-H3 response was 61 percent.", "one.md"), 0.9, 0.9, 0),
        SearchHit(Document("two", "Two", "B7-H3 response needs validation.", "two.md"), 0.8, 0.8, 0),
    ]

    answer = SummaryAgent().run("What was the B7-H3 response?", hits)

    assert "[1]" in answer
    assert "[2]" in answer

