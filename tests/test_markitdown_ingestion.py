from biorag.markitdown_ingestion import MarkItDownIngestor


def test_markitdown_ingests_common_document_as_provenance_marked_text(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Trial\n\nThe endpoint improved.", encoding="utf-8")
    documents = MarkItDownIngestor().ingest(source)
    assert "endpoint improved" in documents[0].text
    assert documents[0].metadata["parser"] == "markitdown"
