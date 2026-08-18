"""Ingest a local PDF corpus and write truthful index metrics to JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from biorag.service import BioRAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BioRAG ingestion on a local PDF corpus.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/corpus_metrics.json"))
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()

    pdfs = sorted(args.corpus.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.corpus}")
    index_path = args.output.parent / "benchmark_index.json"
    service = BioRAGService(index_path, production=args.production)
    started = time.perf_counter()
    result = service.ingest(pdfs)
    elapsed_seconds = round(time.perf_counter() - started, 3)
    report = {
        "mode": "production" if args.production else "offline",
        "input_pdfs": len(pdfs),
        "ingestion_result": result,
        "metrics": service.metrics(),
        "elapsed_seconds": elapsed_seconds,
        "pdfs_per_second": round(len(pdfs) / elapsed_seconds, 3) if elapsed_seconds else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
