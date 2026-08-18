from __future__ import annotations

import argparse
from pathlib import Path

from .service import BioRAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="BioRAG biomedical retrieval")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use Gemini embeddings/generation and FAISS instead of the deterministic offline mode.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("paths", nargs="+", type=Path)
    ask = subparsers.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    subparsers.add_parser("demo")
    args = parser.parse_args()

    production = getattr(args, "production", False)
    service = BioRAGService(Path("data/index/index.json"), production=production)
    if args.command == "ingest":
        print(service.ingest(args.paths))
    elif args.command == "ask":
        result = service.ask(args.question, args.top_k)
        print(result.answer)
        for index, citation in enumerate(result.citations, start=1):
            print(f"[{index}] {citation}")
    else:
        paths = [
            Path("data/demo/biomarkers.md"),
            Path("data/demo/trial.figures.json"),
            Path("data/demo/supplementary.csv"),
        ]
        print("Index:", service.ingest(paths))
        result = service.ask("How does the biomarker affect treatment response?")
        print("\nAnswer:", result.answer)
        for index, citation in enumerate(result.citations, start=1):
            print(f"[{index}] {citation}")
        print("\nAgent trace:")
        for item in result.trace:
            print(f"- {item.agent}: {item.action} ({item.detail})")


if __name__ == "__main__":
    main()
