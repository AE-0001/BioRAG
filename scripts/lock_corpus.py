"""Create a reproducible lock file for the downloaded benchmark corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_lock(manifest_path: Path, pdf_dir: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    for paper in manifest["papers"]:
        matches = sorted(pdf_dir.glob(f"{paper['pmcid']}.*.pdf"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one PDF for {paper['pmcid']}; found {len(matches)}"
            )
        path = matches[0]
        files.append(
            {
                "pmcid": paper["pmcid"],
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "license": paper["license"],
                "split": paper["split"],
            }
        )
    return {
        "schema_version": 1,
        "corpus_id": manifest["corpus_id"],
        "snapshot_date": manifest["snapshot_date"],
        "generated_at": datetime.now(UTC).isoformat(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/manifest.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/corpus/pdfs"))
    parser.add_argument("--output", type=Path, default=Path("data/corpus/corpus.lock.json"))
    args = parser.parse_args()
    payload = build_lock(args.manifest, args.pdf_dir)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Locked {len(payload['files'])} PDFs in {args.output}")


if __name__ == "__main__":
    main()
