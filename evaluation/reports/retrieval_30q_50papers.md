# BioRAG embedding retrieval benchmark v1

Generated: 2026-09-19T16:12:06.287807+00:00
Corpus: 50 PDFs, 1046 pages, 1318 identical text chunks
Queries: 30; top-k: 5
PyMuPDF extraction time: 5.641 s

| Embeddings | Hit@K | MRR@K | nDCG@K | Indexing (s) | Query p50 (s) | Query p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| ollama | 0.733 | 0.661 | 0.680 | 1269.592 | 2.224 | 2.333 |

## Method

All PDFs are a fixed CC BY PMC snapshot. PyMuPDF text extraction runs once; both providers receive the exact same chunks. Relevance is a source-level PMCID match.

## Limitations

- This is a small, author-curated known-item retrieval benchmark, not a clinical QA benchmark.
- Questions were authored from paper abstracts; page-level answer entailment is not scored.
- Network and machine conditions affect timing; repeat runs are needed for stable latency claims.
- Generation, citation faithfulness, and abstention are evaluated separately.
