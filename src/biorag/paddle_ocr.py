from __future__ import annotations

from pathlib import Path
from typing import Any


def _find_recognized_text(value: Any) -> list[str]:
    """Extract PaddleOCR 3.x recognition output without depending on display helpers."""
    if isinstance(value, dict):
        output: list[str] = []
        for key, child in value.items():
            if key in {"rec_texts", "text"} and isinstance(child, list):
                output.extend(str(item) for item in child if str(item).strip())
            else:
                output.extend(_find_recognized_text(child))
        return output
    if isinstance(value, (list, tuple)):
        return [text for child in value for text in _find_recognized_text(child)]
    return []


class PaddleOcrFallback:
    """PP-OCRv5 fallback for pages where layout parsing yields no usable text."""

    def __init__(self):
        from paddleocr import PaddleOCR

        self.pipeline = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv5",
            device="cpu",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
        )

    def recognize(self, image_path: Path) -> str:
        lines: list[str] = []
        for result in self.pipeline.predict(input=str(image_path)):
            payload = getattr(result, "json", result)
            lines.extend(_find_recognized_text(payload))
        return "\n".join(dict.fromkeys(lines))

