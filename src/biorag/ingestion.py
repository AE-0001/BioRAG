from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Document


def stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def ingest_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [
        Document(
            id=stable_id(str(path.resolve()), text),
            title=path.stem.replace("_", " ").title(),
            text=text,
            source=path.name,
        )
    ]


def ingest_pdf(path: Path) -> list[Document]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires PyMuPDF: pip install pymupdf") from exc

    documents: list[Document] = []
    with fitz.open(path) as pdf:
        title = pdf.metadata.get("title") or path.stem.replace("_", " ").title()
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            used_ocr = False
            if len(text) < 40:
                try:
                    text_page = page.get_textpage_ocr(language="eng", dpi=200, full=True)
                    text = page.get_text("text", textpage=text_page).strip()
                    used_ocr = True
                except RuntimeError:
                    # Tesseract is an optional system dependency; preserve any native text.
                    pass
            if text:
                documents.append(
                    Document(
                        id=stable_id(str(path.resolve()), str(page_number), text),
                        title=title,
                        text=text,
                        source=path.name,
                        page=page_number,
                        metadata={"ocr": used_ocr, "kind": "paper_text"},
                    )
                )
            for figure_number, image in enumerate(page.get_images(full=True), start=1):
                xref = image[0]
                image_info = pdf.extract_image(xref)
                width, height = image_info.get("width", 0), image_info.get("height", 0)
                if width < 150 or height < 100:
                    continue
                caption = _nearest_figure_caption(page, figure_number)
                documents.append(
                    Document(
                        id=stable_id(str(path.resolve()), str(page_number), str(xref)),
                        title=f"{title} - Figure {figure_number}",
                        text=caption or f"Extracted biomedical figure {figure_number} on page {page_number}.",
                        source=path.name,
                        modality="figure",
                        page=page_number,
                        metadata={
                            "paper": title,
                            "kind": "extracted_figure",
                            "xref": xref,
                            "width": width,
                            "height": height,
                            "image_extension": image_info.get("ext"),
                        },
                    )
                )
    return documents


def _nearest_figure_caption(page: Any, figure_number: int) -> str:
    lines = [line.strip() for line in page.get_text("text").splitlines()]
    candidates = [
        line
        for line in lines
        if line.lower().startswith(("figure ", "fig. ", "fig "))
    ]
    return candidates[min(figure_number - 1, len(candidates) - 1)] if candidates else ""


def ingest_figure_manifest(path: Path) -> list[Document]:
    """Ingest human/model-generated figure captions as retrievable visual evidence."""
    items = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            id=stable_id(item["image"], item["caption"]),
            title=item.get("title", Path(item["image"]).stem),
            text=" ".join(filter(None, [item.get("caption"), item.get("ocr_text")])),
            source=item["image"],
            modality="figure",
            page=item.get("page"),
            metadata={"paper": item.get("paper"), "kind": item.get("kind", "figure")},
        )
        for item in items
    ]


def ingest_dataset(path: Path) -> list[Document]:
    """Turn supplementary tabular rows into provenance-preserving evidence."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Dataset ingestion requires pandas") from exc

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.json_normalize(records)
    frame = frame.fillna("")
    documents: list[Document] = []
    for row_number, row in frame.iterrows():
        values = "; ".join(f"{column}: {value}" for column, value in row.items())
        documents.append(
            Document(
                id=stable_id(str(path.resolve()), str(row_number), values),
                title=f"{path.stem} supplementary data row {row_number + 1}",
                text=values,
                source=path.name,
                metadata={"kind": "supplementary_dataset", "row": int(row_number) + 1},
            )
        )
    return documents


def ingest_path(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return ingest_text(path)
    if suffix == ".pdf":
        return ingest_pdf(path)
    if suffix == ".json" and path.name.endswith(".figures.json"):
        return ingest_figure_manifest(path)
    if suffix in {".csv", ".tsv", ".xlsx", ".xls", ".json"}:
        return ingest_dataset(path)
    raise ValueError(
        f"Unsupported input: {path}. Use PDF, TXT, MD, CSV, TSV, XLSX, JSON, "
        "or *.figures.json."
    )

