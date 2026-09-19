from pathlib import Path

from biorag.graph import FourAgentResearchGraph
from biorag.models import Document, SearchHit


def hit(identifier="1", modality="text", text="Supported finding.", image_path=None):
    metadata = {"image_path": str(image_path)} if image_path else {}
    document = Document(
        identifier, "Study", text, "study.pdf", modality=modality, page=4, metadata=metadata
    )
    return SearchHit(document, 0.9, 0.8, 0.9)


class StaticRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.hits[:top_k]


class ImprovingRetriever:
    def __init__(self):
        self.queries = []

    def search(self, question, top_k=5):
        self.queries.append(question)
        return [] if len(self.queries) == 1 else [hit()]


class FakeGenerator:
    def __init__(self, answer="Supported finding. [1]"):
        self.generated_answer = answer
        self.inspected = []
        self.questions = []

    def answer(self, question, contexts):
        self.questions.append(question)
        return self.generated_answer

    def inspect_figure(self, image_path: Path, caption: str, question: str):
        self.inspected.append((image_path, caption, question))
        return "The response bar is higher."


def test_retrieval_receives_question_and_top_k():
    retriever = StaticRetriever([hit()])
    FourAgentResearchGraph(retriever).invoke("What is supported?", top_k=1)
    assert retriever.calls == [("What is supported?", 1)]


def test_follow_up_uses_recent_conversation_for_retrieval_and_generation():
    retriever = StaticRetriever([hit()])
    generator = FakeGenerator()
    history = [{"question": "What mechanisms are reported?", "answer": "Efflux pumps."}]
    FourAgentResearchGraph(retriever, generator).invoke(
        "How do they work?", top_k=1, history=history
    )
    assert "Previous topic: What mechanisms are reported?" in retriever.calls[0][0]
    assert "Conversation context:" in generator.questions[0]
    assert "Previous answer: Efflux pumps." in generator.questions[0]


def test_no_evidence_sets_warning_and_not_grounded():
    result = FourAgentResearchGraph(StaticRetriever([])).invoke("Unknown question")
    assert result["grounded"] is False
    assert "insufficient_evidence" in result["warnings"]
    assert "citation_verification_failed" in result["warnings"]


def test_invalid_citation_number_preserves_answer_for_review():
    generator = FakeGenerator("Unsupported index. [2]")
    result = FourAgentResearchGraph(StaticRetriever([hit()]), generator).invoke("Question")
    assert result["grounded"] is False
    assert result["answer"] == "Unsupported index. [2]"
    assert "citation_verification_failed" in result["warnings"]


def test_missing_citation_preserves_answer_for_review():
    generator = FakeGenerator("A claim without evidence marker.")
    result = FourAgentResearchGraph(StaticRetriever([hit()]), generator).invoke("Question")
    assert result["grounded"] is False
    assert result["answer"] == "A claim without evidence marker."
    assert "citation_verification_failed" in result["warnings"]


def test_valid_multiple_citations_pass_verification():
    generator = FakeGenerator("First claim. [1] Second claim. [2]")
    result = FourAgentResearchGraph(StaticRetriever([hit("1"), hit("2")]), generator).invoke(
        "Question"
    )
    assert result["grounded"] is True


def test_figure_is_promoted_ahead_of_text():
    retriever = StaticRetriever([hit("text"), hit("figure", modality="figure")])
    result = FourAgentResearchGraph(retriever).invoke("Compare the figure")
    assert result["hits"][0].document.modality == "figure"


def test_existing_figure_is_inspected_by_generator(tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"not-decoded-by-fake-generator")
    generator = FakeGenerator()
    result = FourAgentResearchGraph(
        StaticRetriever([hit("figure", modality="figure", image_path=image)]), generator
    ).invoke("What does the figure show?")
    assert len(generator.inspected) == 1
    assert result["hits"][0].document.metadata["vision_analyzed"] is True
    assert "Visual analysis" in result["hits"][0].document.text


def test_missing_figure_file_is_not_sent_to_generator(tmp_path):
    generator = FakeGenerator()
    missing = tmp_path / "missing.png"
    FourAgentResearchGraph(
        StaticRetriever([hit("figure", modality="figure", image_path=missing)]), generator
    ).invoke("Question")
    assert generator.inspected == []


def test_trace_records_conditional_non_visual_route():
    result = FourAgentResearchGraph(StaticRetriever([hit()])).invoke("Question")
    assert [step["agent"] for step in result["trace"]] == [
        "retrieval",
        "evidence_grader",
        "research_summary",
        "citation",
    ]


def test_low_relevance_route_rewrites_and_retries_once():
    retriever = ImprovingRetriever()
    result = FourAgentResearchGraph(retriever).invoke("What is supported?")
    assert result["grounded"] is True
    assert result["retrieval_attempts"] == 2
    assert "biomedical study evidence" in retriever.queries[1]
    assert "query_rewriter" in [step["agent"] for step in result["trace"]]
