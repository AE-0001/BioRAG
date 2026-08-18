from biorag.service import BioRAGService


def test_reingestion_is_idempotent(tmp_path, demo_paper):
    service = BioRAGService(tmp_path / "index" / "index.json")
    first = service.ingest([demo_paper])
    second = service.ingest([demo_paper])
    assert first["chunks"] > 0
    assert second["chunks"] == 0
    assert second["indexed_total"] == first["indexed_total"]


def test_index_survives_service_restart(tmp_path, demo_paper):
    path = tmp_path / "index" / "index.json"
    first = BioRAGService(path)
    first.ingest([demo_paper])
    restored = BioRAGService(path)
    assert restored.search("B7-H3", 1)[0].document.source == "trial.md"


def test_metrics_reflect_supplementary_rows(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("group,value\na,1\nb,2\n", encoding="utf-8")
    service = BioRAGService(tmp_path / "index.json")
    service.ingest([dataset])
    assert service.metrics()["supplementary_rows"] == 2


def test_offline_backend_is_reported_truthfully(tmp_path):
    service = BioRAGService(tmp_path / "index.json")
    assert service.metrics()["backend"] == "offline-test-double"


def test_ingest_reports_document_and_chunk_counts(tmp_path, demo_paper):
    service = BioRAGService(tmp_path / "index.json")
    result = service.ingest([demo_paper])
    assert result["documents"] == 1
    assert result["chunks"] == result["indexed_total"]


def test_answer_has_source_citation(tmp_path, demo_paper):
    service = BioRAGService(tmp_path / "index.json")
    service.ingest([demo_paper])
    result = service.ask("What was associated with response?")
    assert result.grounded is True
    assert any("trial.md" in citation for citation in result.citations)


def test_search_obeys_requested_limit(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("Shared biomarker evidence.", encoding="utf-8")
    second.write_text("Shared treatment evidence.", encoding="utf-8")
    service = BioRAGService(tmp_path / "index.json")
    service.ingest([first, second])
    assert len(service.search("shared", top_k=1)) == 1

