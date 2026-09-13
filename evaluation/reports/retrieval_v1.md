# BioRAG embedding retrieval benchmark v1

Generated: 2026-09-13T14:12:47.863460+00:00
Corpus: 3 PDFs, 67 pages, 85 identical text chunks
Queries: 6; top-k: 5
PyMuPDF extraction time: 0.282 s

| Embeddings | Hit@K | MRR@K | nDCG@K | Indexing (s) | Query p50 (s) | Query p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| ollama:nomic-embed-text | 1.000 | 1.000 | 1.000 | 77.903 | 2.267 | 2.336 |
| Gemini Embedding 2 | not reported | not reported | not reported | quota blocked | — | — |

## Method

All PDFs are a fixed CC BY PMC snapshot. PyMuPDF text extraction runs once; both providers receive the exact same chunks. Relevance is a source-level PMCID match.

The Gemini arm was attempted with both `gemini-embedding-2` and
`gemini-embedding-001`. Google returned HTTP 429 `RESOURCE_EXHAUSTED` for the
project's free embedding quota, so no Gemini scores are invented or inferred from
the earlier two-vector connectivity smoke test. Re-running the committed command
after quota reset completes the missing arm.

## Limitations

- This is a small, author-curated known-item retrieval benchmark, not a clinical QA benchmark.
- Questions were authored from paper abstracts; page-level answer entailment is not scored.
- Network and machine conditions affect timing; repeat runs are needed for stable latency claims.
- Generation, citation faithfulness, and abstention are evaluated separately.
