from biorag.graph import FourAgentResearchGraph
from biorag.models import Document, SearchHit


class FakeRetriever:
    def search(self, query, top_k=5):
        document = Document(
            id="paper:1",
            title="Controlled study",
            text="The observed biomarker association requires prospective validation.",
            source="paper.pdf",
            page=7,
        )
        return [SearchHit(document, 0.9, 0.8, 0.9)]


def test_compiled_graph_runs_four_agents():
    result = FourAgentResearchGraph(FakeRetriever()).invoke("What is the limitation?", 1)
    assert result["grounded"] is True
    assert result["citations"] == ["Controlled study (paper.pdf, p. 7)"]
    assert [step["agent"] for step in result["trace"]] == [
        "retrieval",
        "evidence_grader",
        "research_summary",
        "citation",
    ]
