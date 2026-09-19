# BioRAG embedding retrieval benchmark v1

Generated: 2026-09-19T15:26:16.243636+00:00
Corpus: 13 PDFs, 245 pages, 329 identical text chunks
Queries: 30; top-k: 5
PyMuPDF extraction time: 1.267 s

| Embeddings | Hit@K | MRR@K | nDCG@K | Indexing (s) | Query p50 (s) | Query p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| ollama | 0.900 | 0.806 | 0.830 | 323.265 | 2.256 | 2.339 |

## Method

All PDFs are a fixed CC BY PMC snapshot. PyMuPDF text extraction runs once; both providers receive the exact same chunks. Relevance is a source-level PMCID match.

## Limitations

- This is a small, author-curated known-item retrieval benchmark, not a clinical QA benchmark.
- Questions were authored from paper abstracts; page-level answer entailment is not scored.
- Network and machine conditions affect timing; repeat runs are needed for stable latency claims.
- Generation, citation faithfulness, and abstention are evaluated separately.
