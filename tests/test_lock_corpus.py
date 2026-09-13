import hashlib
import json

import pytest

from scripts.lock_corpus import build_lock


def test_build_lock_records_identity_and_metadata(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    content = b"%PDF-1.7 test"
    (pdf_dir / "PMC1.2.pdf").write_bytes(content)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpus_id": "test-corpus",
                "snapshot_date": "2026-09-13",
                "papers": [
                    {"pmcid": "PMC1", "license": "CC BY 4.0", "split": "test"}
                ],
            }
        ),
        encoding="utf-8",
    )

    lock = build_lock(manifest, pdf_dir)

    assert lock["corpus_id"] == "test-corpus"
    assert lock["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    assert lock["files"][0]["filename"] == "PMC1.2.pdf"


def test_build_lock_rejects_missing_pdf(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpus_id": "test",
                "snapshot_date": "2026-09-13",
                "papers": [{"pmcid": "PMC404", "license": "CC BY", "split": "test"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exactly one PDF"):
        build_lock(manifest, tmp_path)
