from fastapi.testclient import TestClient

from biorag import api
from biorag.service import BioRAGService


def client_with_empty_service(tmp_path, monkeypatch):
    service = BioRAGService(tmp_path / "index" / "index.json")
    monkeypatch.setattr(api, "service", service)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    return TestClient(api.app)


def test_health_contract(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "offline", "indexed_chunks": 0}


def test_metrics_contract_has_stable_keys(tmp_path, monkeypatch):
    payload = client_with_empty_service(tmp_path, monkeypatch).get("/stats").json()
    assert set(payload) == {
        "backend",
        "papers",
        "figures",
        "tables",
        "supplementary_rows",
        "chunks",
    }


def test_prometheus_metrics_use_exposition_format(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "biorag_queries_total" in response.text


def test_question_rejects_too_short_input(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).post(
        "/ask", json={"question": "?", "top_k": 5}
    )
    assert response.status_code == 422


def test_question_rejects_zero_top_k(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).post(
        "/ask", json={"question": "What changed?", "top_k": 0}
    )
    assert response.status_code == 422


def test_question_rejects_excessive_top_k(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).post(
        "/ask", json={"question": "What changed?", "top_k": 21}
    )
    assert response.status_code == 422


def test_empty_index_returns_insufficient_evidence(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).post(
        "/ask", json={"question": "What changed?", "top_k": 5}
    )
    assert response.status_code == 200
    assert "sufficient evidence" in response.json()["answer"].lower()
    assert response.json()["grounded"] is False


def test_search_rejects_invalid_top_k(tmp_path, monkeypatch):
    client = client_with_empty_service(tmp_path, monkeypatch)
    assert client.get("/search", params={"q": "marker", "top_k": 0}).status_code == 422
    assert client.get("/search", params={"q": "marker", "top_k": 21}).status_code == 422


def test_search_empty_index_has_no_hits(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).get(
        "/search", params={"q": "B7-H3", "top_k": 3}
    )
    assert response.status_code == 200
    assert response.json() == {"query": "B7-H3", "hits": []}


def test_ingest_missing_path_returns_404(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).post(
        "/ingest", json={"paths": [str(tmp_path / "missing.pdf")]}
    )
    assert response.status_code == 404
    assert response.json()["detail"]["missing"] == [str(tmp_path / "missing.pdf")]


def test_upload_rejects_unsupported_extension(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).post(
        "/documents/upload", files=[("files", ("payload.exe", b"bad", "application/octet-stream"))]
    )
    assert response.status_code == 415


def test_upload_strips_path_traversal_from_filename(tmp_path, monkeypatch):
    response = client_with_empty_service(tmp_path, monkeypatch).post(
        "/documents/upload",
        files=[("files", ("../../paper.md", b"Biomedical evidence.", "text/markdown"))],
    )
    assert response.status_code == 200
    saved = response.json()["files"][0].replace("\\", "/")
    assert saved.endswith("/paper.md")
    assert ".." not in saved


def test_upload_and_ask_round_trip(tmp_path, monkeypatch):
    client = client_with_empty_service(tmp_path, monkeypatch)
    uploaded = client.post(
        "/documents/upload",
        files=[("files", ("paper.md", b"B7-H3 predicted treatment response.", "text/markdown"))],
    )
    assert uploaded.status_code == 200
    answer = client.post(
        "/ask", json={"question": "What predicted treatment response?", "top_k": 3}
    )
    assert answer.status_code == 200
    assert "B7-H3" in answer.json()["answer"]
    assert answer.json()["citations"]
