from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .service import BioRAGService

DATA_DIR = Path(os.getenv("BIORAG_DATA_DIR", "data"))
PRODUCTION = os.getenv("BIORAG_MODE", "offline").lower() == "production"
INDEX_PATH = Path(os.getenv("BIORAG_INDEX_PATH", str(DATA_DIR / "index" / "index.json")))
UPLOAD_FILES = File(...)
service = BioRAGService(INDEX_PATH, production=PRODUCTION)
app = FastAPI(
    title="BioRAG API",
    version="0.1.0",
    description="Evidence-grounded biomedical literature retrieval. Not for clinical use.",
)


class IngestRequest(BaseModel):
    paths: list[str] = Field(min_length=1)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health() -> dict[str, int | str]:
    return {
        "status": "ok",
        "mode": "production" if service.production else "offline",
        "indexed_chunks": len(service.index.documents),
    }


@app.get("/stats")
def stats() -> dict[str, int | str]:
    return service.metrics()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/documents/upload")
def upload_documents(files: list[UploadFile] = UPLOAD_FILES) -> dict:
    upload_dir = DATA_DIR / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    allowed = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".tsv", ".xlsx", ".xls", ".json"}
    for upload in files:
        filename = Path(upload.filename or "").name
        if Path(filename).suffix.lower() not in allowed:
            raise HTTPException(status_code=415, detail=f"Unsupported file: {filename}")
        destination = upload_dir / filename
        with destination.open("wb") as stream:
            shutil.copyfileobj(upload.file, stream)
        saved.append(destination)
    return {"files": [str(path) for path in saved], "result": service.ingest(saved)}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, int]:
    paths = [Path(path) for path in request.paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise HTTPException(status_code=404, detail={"missing": missing})
    try:
        return service.ingest(paths)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: QuestionRequest) -> dict:
    result = service.ask(request.question, request.top_k)
    if isinstance(result, dict):
        return {
            **result,
            "evidence": [
                {
                    "title": hit.document.title,
                    "source": hit.document.source,
                    "page": hit.document.page,
                    "modality": hit.document.modality,
                    "score": round(hit.score, 4),
                    "text": hit.document.text,
                    "metadata": hit.document.metadata,
                }
                for hit in result.get("hits", [])
            ],
            "hits": None,
        }
    return {
        "question": result.question,
        "answer": result.answer,
        "citations": result.citations,
        "grounded": result.grounded,
        "evidence": [
            {
                "title": hit.document.title,
                "source": hit.document.source,
                "page": hit.document.page,
                "modality": hit.document.modality,
                "score": round(hit.score, 4),
                "text": hit.document.text,
            }
            for hit in result.evidence
        ],
        "trace": [vars(item) for item in result.trace],
    }


@app.get("/search")
def search(q: str, top_k: int = 5) -> dict:
    if not 1 <= top_k <= 20:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 20")
    return {
        "query": q,
        "hits": [
            {
                "title": hit.document.title,
                "source": hit.document.source,
                "page": hit.document.page,
                "modality": hit.document.modality,
                "score": round(hit.score, 5),
                "lexical_score": round(hit.lexical_score, 5),
                "semantic_score": round(hit.semantic_score, 5),
                "text": hit.document.text,
            }
            for hit in service.search(q, top_k)
        ],
    }
