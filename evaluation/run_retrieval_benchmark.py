from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import fitz
from metrics import ndcg, percentile, reciprocal_rank

from biorag.chunking import chunk_documents
from biorag.faiss_store import FaissStore
from biorag.gemini import GeminiEmbeddings
from biorag.ingestion import stable_id
from biorag.local_ai import OllamaEmbeddings
from biorag.models import Document


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extract_text_chunks(pdfs: list[Path]) -> tuple[list[Document], float]:
    started = time.perf_counter()
    pages: list[Document] = []
    for path in pdfs:
        with fitz.open(path) as pdf:
            title = pdf.metadata.get("title") or path.stem
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append(
                        Document(
                            id=stable_id(path.name, str(page_number), text),
                            title=title,
                            text=text,
                            source=path.name,
                            page=page_number,
                            metadata={"kind": "paper_text", "parser": "pymupdf"},
                        )
                    )
    # Larger evidence windows keep this portfolio benchmark affordable while
    # retaining page provenance and identical inputs across both providers.
    return chunk_documents(pages, max_chars=6000), time.perf_counter() - started


def evaluate(name: str, embeddings, chunks: list[Document], cases: list[dict], top_k: int):
    store = FaissStore(embeddings)
    started = time.perf_counter()
    store.add(chunks)
    indexing_seconds = time.perf_counter() - started
    records = []
    for case in cases:
        started = time.perf_counter()
        hits = store.search(case["question"], top_k)
        latency = time.perf_counter() - started
        ranked_sources = list(dict.fromkeys(hit.document.source for hit in hits))
        relevant = [
            any(pmcid.lower() in source.lower() for pmcid in case["gold_pmcids"])
            for source in ranked_sources
        ]
        records.append(
            {
                "id": case["id"],
                "hit_at_k": any(relevant),
                "reciprocal_rank": reciprocal_rank(relevant),
                "ndcg": ndcg(relevant, len(case["gold_pmcids"])),
                "latency_seconds": latency,
                "retrieved": [
                    {
                        "source": hit.document.source,
                        "page": hit.document.page,
                        "score": round(hit.score, 6),
                    }
                    for hit in hits
                ],
            }
        )
    latencies = [record["latency_seconds"] for record in records]
    return {
        "provider": name,
        "indexing_seconds": indexing_seconds,
        "aggregate": {
            "questions": len(records),
            "hit_at_k": statistics.mean(record["hit_at_k"] for record in records),
            "mrr_at_k": statistics.mean(record["reciprocal_rank"] for record in records),
            "ndcg_at_k": statistics.mean(record["ndcg"] for record in records),
            "query_p50_seconds": percentile(latencies, 0.5),
            "query_p95_seconds": percentile(latencies, 0.95),
        },
        "cases": records,
    }


def corpus_digest(pdfs: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in pdfs:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# BioRAG embedding retrieval benchmark v1",
        "",
        f"Generated: {report['generated_at']}",
        (
            f"Corpus: {report['corpus']['pdfs']} PDFs, {report['corpus']['pages']} pages, "
            f"{report['corpus']['chunks']} identical text chunks"
        ),
        f"Queries: {report['queries']}; top-k: {report['top_k']}",
        f"PyMuPDF extraction time: {report['extraction_seconds']:.3f} s",
        "",
        "| Embeddings | Hit@K | MRR@K | nDCG@K | Indexing (s) | Query p50 (s) | Query p95 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        metric = run["aggregate"]
        lines.append(
            f"| {run['provider']} | {metric['hit_at_k']:.3f} | {metric['mrr_at_k']:.3f} | "
            f"{metric['ndcg_at_k']:.3f} | {run['indexing_seconds']:.3f} | "
            f"{metric['query_p50_seconds']:.3f} | {metric['query_p95_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            (
                "All PDFs are a fixed CC BY PMC snapshot. PyMuPDF text extraction runs once; "
                "both providers receive the exact same chunks. Relevance is a source-level "
                "PMCID match."
            ),
            "",
            "## Limitations",
            "",
            "- This is a small, author-curated known-item retrieval benchmark, not a clinical QA benchmark.",
            "- Questions were authored from paper abstracts; page-level answer entailment is not scored.",
            "- Network and machine conditions affect timing; repeat runs are needed for stable latency claims.",
            "- Generation, citation faithfulness, and abstention are evaluated separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/pdfs"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("evaluation/datasets/biorag_retrieval_v1.jsonl")
    )
    parser.add_argument("--providers", nargs="+", choices=("ollama", "gemini"), default=("ollama", "gemini"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pmcids", nargs="*", default=[])
    parser.add_argument("--gemini-model", default="gemini-embedding-2")
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/retrieval_v1.json"))
    args = parser.parse_args()
    pdfs = sorted(args.corpus.glob("*.pdf"))
    if args.pmcids:
        wanted = {item.lower() for item in args.pmcids}
        pdfs = [path for path in pdfs if any(item in path.name.lower() for item in wanted)]
    if not pdfs:
        raise SystemExit("No PDFs found")
    cases = load_cases(args.dataset)
    if args.pmcids:
        wanted = {item.lower() for item in args.pmcids}
        cases = [
            case
            for case in cases
            if any(pmcid.lower() in wanted for pmcid in case["gold_pmcids"])
        ]
    chunks, extraction_seconds = extract_text_chunks(pdfs)
    providers = {
        "ollama": lambda: OllamaEmbeddings(model="nomic-embed-text"),
        "gemini": lambda: GeminiEmbeddings(model=args.gemini_model),
    }
    runs = [evaluate(name, providers[name](), chunks, cases, args.top_k) for name in args.providers]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "corpus": {
            "pdfs": len(pdfs),
            "pages": sum(len(fitz.open(path)) for path in pdfs),
            "chunks": len(chunks),
            "sha256": corpus_digest(pdfs),
        },
        "queries": len(cases),
        "top_k": args.top_k,
        "extraction_seconds": extraction_seconds,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.output.with_suffix(".md"))
    print(args.output)


if __name__ == "__main__":
    main()
