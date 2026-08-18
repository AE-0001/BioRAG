import pytest

from biorag.chunking import chunk_documents, semantic_chunks
from biorag.models import Document


@pytest.mark.parametrize("value", ["", " ", "\n\t", "   \r\n"])
def test_blank_inputs_produce_no_chunks(value):
    assert semantic_chunks(value) == []


def test_whitespace_is_normalized():
    assert semantic_chunks("Alpha   beta.\n\nGamma.") == ["Alpha beta. Gamma."]


def test_overlap_repeats_boundary_sentence():
    chunks = semantic_chunks(
        "First sentence here. Second sentence here. Third sentence here.",
        max_chars=43,
        overlap_sentences=1,
    )
    assert len(chunks) >= 2
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]


def test_zero_overlap_does_not_repeat_content():
    text = "First sentence. Second sentence. Third sentence."
    chunks = semantic_chunks(text, max_chars=24, overlap_sentences=0)
    assert " ".join(chunks) == text


def test_single_long_sentence_is_not_dropped():
    sentence = "biomarker " * 100
    assert semantic_chunks(sentence, max_chars=20) == [sentence.strip()]


def test_document_provenance_survives_chunking():
    source = Document(
        "paper", "Study", "First finding. Second finding.", "study.pdf",
        page=8, metadata={"doi": "10.1/example"}
    )
    chunk = chunk_documents([source], max_chars=20)[0]
    assert chunk.source == "study.pdf"
    assert chunk.page == 8
    assert chunk.metadata["doi"] == "10.1/example"
    assert chunk.metadata["parent_id"] == "paper"


def test_figure_modality_survives_chunking():
    source = Document("fig", "Figure", "Kaplan-Meier curve.", "figure.png", modality="figure")
    assert chunk_documents([source])[0].modality == "figure"


def test_empty_documents_are_omitted():
    source = Document("empty", "Empty", "  ", "paper.pdf")
    assert chunk_documents([source]) == []

