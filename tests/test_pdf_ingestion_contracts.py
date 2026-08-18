import fitz

from biorag.ingestion import ingest_pdf


def make_pdf(path, pages):
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_native_pdf_text_is_extracted_with_page_provenance(tmp_path):
    path = tmp_path / "study.pdf"
    make_pdf(path, ["First biomedical finding with enough native text for extraction."])
    evidence = ingest_pdf(path)
    text = [document for document in evidence if document.modality == "text"]
    assert len(text) == 1
    assert text[0].page == 1
    assert "First biomedical finding" in text[0].text
    assert text[0].metadata["ocr"] is False


def test_multiple_pdf_pages_keep_distinct_page_numbers(tmp_path):
    path = tmp_path / "study.pdf"
    make_pdf(path, [
        "Page one contains a sufficiently long biomedical observation.",
        "Page two contains a sufficiently long independent observation.",
    ])
    evidence = [document for document in ingest_pdf(path) if document.modality == "text"]
    assert [document.page for document in evidence] == [1, 2]


def test_pdf_ids_are_stable_across_repeated_ingestion(tmp_path):
    path = tmp_path / "study.pdf"
    make_pdf(path, ["Stable biomedical content that is sufficiently long for native parsing."])
    assert [doc.id for doc in ingest_pdf(path)] == [doc.id for doc in ingest_pdf(path)]

