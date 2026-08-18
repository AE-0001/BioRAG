import json

import pytest

from biorag.ingestion import ingest_dataset, ingest_path, stable_id


def test_stable_id_is_deterministic_and_compact():
    assert stable_id("paper", "page") == stable_id("paper", "page")
    assert len(stable_id("paper", "page")) == 16


def test_stable_id_changes_when_content_changes():
    assert stable_id("paper", "one") != stable_id("paper", "two")


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "payload.exe"
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported input"):
        ingest_path(path)


def test_csv_rows_preserve_column_names(tmp_path):
    path = tmp_path / "supplement.csv"
    path.write_text("group,response\nhigh,61\nlow,29\n", encoding="utf-8")
    documents = ingest_dataset(path)
    assert len(documents) == 2
    assert "group: high" in documents[0].text
    assert "response: 61" in documents[0].text


def test_tsv_is_parsed_with_tab_separator(tmp_path):
    path = tmp_path / "supplement.tsv"
    path.write_text("group\tresponse\nhigh\t61\n", encoding="utf-8")
    assert "response: 61" in ingest_dataset(path)[0].text


def test_json_records_are_flattened(tmp_path):
    path = tmp_path / "supplement.json"
    path.write_text(json.dumps([{"patient": {"group": "high"}, "response": 61}]), encoding="utf-8")
    document = ingest_dataset(path)[0]
    assert "patient.group: high" in document.text
    assert document.metadata == {"kind": "supplementary_dataset", "row": 1}


def test_malformed_json_is_not_silently_accepted(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        ingest_dataset(path)


def test_missing_values_are_normalized_to_empty_text(tmp_path):
    path = tmp_path / "missing.csv"
    path.write_text("group,response\nhigh,\n", encoding="utf-8")
    assert "response: " in ingest_dataset(path)[0].text


def test_markdown_routes_to_text_ingestion(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text("# Evidence", encoding="utf-8")
    document = ingest_path(path)[0]
    assert document.title == "Paper"
    assert document.text == "# Evidence"

