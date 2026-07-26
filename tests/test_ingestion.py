import json

from biorag.ingestion import ingest_figure_manifest, ingest_text


def test_text_ingestion(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text("Biomedical evidence.", encoding="utf-8")
    documents = ingest_text(path)
    assert documents[0].text == "Biomedical evidence."
    assert documents[0].modality == "text"


def test_figure_manifest_is_visual_evidence(tmp_path):
    path = tmp_path / "paper.figures.json"
    path.write_text(
        json.dumps([{"image": "f1.png", "caption": "A microscopy image.", "ocr_text": "CD8"}]),
        encoding="utf-8",
    )
    document = ingest_figure_manifest(path)[0]
    assert document.modality == "figure"
    assert "CD8" in document.text

