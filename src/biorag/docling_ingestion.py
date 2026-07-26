from __future__ import annotations

import hashlib
from pathlib import Path

from .models import Document
from .paddle_ocr import PaddleOcrFallback


class DoclingPaperIngestor:
    """Layout-aware scientific PDF parser with PaddleOCR-backed scanned-page support."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir

    def ingest(self, path: Path) -> list[Document]:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise RuntimeError("Install project dependencies to use Docling ingestion") from exc

        options = PdfPipelineOptions()
        options.do_ocr = True
        options.do_table_structure = True
        options.generate_picture_images = True
        options.images_scale = 2.0
        options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.AUTO,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        result = converter.convert(path)
        paper_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        paper_dir = self.artifacts_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        title = result.document.name or path.stem
        documents: list[Document] = []

        for index, item in enumerate(result.document.texts):
            text = (item.text or "").strip()
            if not text:
                continue
            provenance = item.prov[0] if item.prov else None
            page = provenance.page_no if provenance else None
            label = str(getattr(item, "label", "text"))
            documents.append(
                Document(
                    id=f"{paper_id}:text:{index}",
                    title=title,
                    text=text,
                    source=path.name,
                    page=page,
                    metadata={
                        "paper_id": paper_id,
                        "kind": "paper_text",
                        "label": label,
                        "parser": "docling",
                    },
                )
            )

        for index, table in enumerate(result.document.tables):
            provenance = table.prov[0] if table.prov else None
            page = provenance.page_no if provenance else None
            markdown = table.export_to_markdown(doc=result.document)
            documents.append(
                Document(
                    id=f"{paper_id}:table:{index}",
                    title=f"{title} - Table {index + 1}",
                    text=markdown,
                    source=path.name,
                    page=page,
                    metadata={"paper_id": paper_id, "kind": "table", "parser": "docling"},
                )
            )

        for index, picture in enumerate(result.document.pictures):
            provenance = picture.prov[0] if picture.prov else None
            page = provenance.page_no if provenance else None
            caption = picture.caption_text(doc=result.document) or ""
            image_path = paper_dir / f"figure-{index + 1}.png"
            image = picture.get_image(result.document)
            if image is not None:
                image.save(image_path)
            documents.append(
                Document(
                    id=f"{paper_id}:figure:{index}",
                    title=f"{title} - Figure {index + 1}",
                    text=caption or f"Scientific figure {index + 1} from {title}.",
                    source=path.name,
                    modality="figure",
                    page=page,
                    metadata={
                        "paper_id": paper_id,
                        "kind": "figure",
                        "parser": "docling",
                        "image_path": str(image_path),
                    },
                )
            )
        pages_with_text = {document.page for document in documents if document.text.strip()}
        self._add_paddle_fallback(path, paper_id, title, pages_with_text, documents, paper_dir)
        return documents

    def _add_paddle_fallback(
        self,
        path: Path,
        paper_id: str,
        title: str,
        pages_with_text: set[int | None],
        documents: list[Document],
        paper_dir: Path,
    ) -> None:
        """Run the expensive OCR model only on pages Docling could not recover."""
        import fitz

        missing: list[tuple[int, Path]] = []
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                if page_number in pages_with_text:
                    continue
                image_path = paper_dir / f"ocr-page-{page_number}.png"
                page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(image_path)
                missing.append((page_number, image_path))
        if not missing:
            return
        ocr = PaddleOcrFallback()
        for page_number, image_path in missing:
            text = ocr.recognize(image_path).strip()
            if text:
                documents.append(
                    Document(
                        id=f"{paper_id}:paddle-ocr:{page_number}",
                        title=title,
                        text=text,
                        source=path.name,
                        page=page_number,
                        metadata={
                            "paper_id": paper_id,
                            "kind": "paper_text",
                            "parser": "paddleocr-v5-fallback",
                        },
                    )
                )
