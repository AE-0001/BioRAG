# BioRAG interview guide

## 30-second pitch

“BioRAG is a multimodal research assistant for biomedical literature. It parses
papers with layout and OCR awareness, preserves page-level provenance for text,
tables, figures, and supplementary data, and combines BM25 with Gemini
embeddings in FAISS. A four-node LangGraph retrieves evidence, aligns visual
context, synthesizes a constrained research answer, and validates citations.
The system exposes its evidence and execution trace through FastAPI and
Streamlit, and I evaluate retrieval and grounding separately.”

## LangChain versus LangGraph

LangChain supplies standardized interfaces for models, embeddings, documents,
retrievers, tools, and prompt composition. LangGraph is the orchestration
runtime: nodes read and update typed shared state, edges define execution, and a
compiled graph can support tracing, persistence, retries, and human review.

BioRAG does not use four agents merely as personas. Each node owns a measurable
contract:

1. Retrieval maximizes evidence recall.
2. Vision connects image evidence to document context.
3. Research Summary minimizes unsupported synthesis.
4. Citation validates claim-to-source references.

## Questions to be ready for

### Why hybrid retrieval?

Dense retrieval captures semantic similarity, while BM25 is strong for exact
gene names, drug identifiers, abbreviations, and rare biomedical terms.
Reciprocal-rank fusion combines rankings without assuming their scores are
calibrated.

### Why FAISS?

It is fast, local, reproducible, and sufficient for tens of thousands of
chunks. The current exact inner-product index provides a reliable benchmark.
For millions of chunks, benchmark HNSW or IVF and measure the recall/latency
trade-off.

### Why not send entire PDFs to Gemini?

Selective retrieval reduces context cost, exposes provenance, supports
repeatable evaluation, and allows access controls at document/chunk level.

### Why Docling and PaddleOCR?

OCR alone returns characters. Scientific RAG also needs reading order, section
boundaries, table structure, captions, figures, and page provenance. Docling
provides document structure; PaddleOCR handles difficult raster text; PyMuPDF
provides robust PDF primitives.

### What makes the answer grounded?

The generator receives only numbered evidence and must cite its claims. The
Citation Agent rejects missing or invalid markers. Stronger production
grounding would decompose the answer into claims and run an entailment check
against each cited passage.

### Why four agents rather than one chain?

They isolate failure domains and evaluation. Retrieval quality, visual
alignment, synthesis, and citation correctness can be tested independently.
If the workflow never needs branching, retries, or state inspection, a simple
chain would be more appropriate.

### What would you improve next?

- Cross-encoder reranking trained on biomedical relevance
- Claim-level entailment and contradiction detection
- PubMed/PMC metadata and DOI normalization
- LangGraph checkpointing and human approval for low-confidence answers
- HNSW/IVF benchmarking at larger scale
- OpenTelemetry or LangSmith traces and latency dashboards

## Honesty rule

Never say the bundled repository contains 500 papers. Say the pipeline was
designed and benchmarked on that corpus only when `/metrics` and the benchmark
artifact demonstrate it. Clearly distinguish the implementation, the demo
corpus, and any previous private benchmark corpus.
# Upgrade talking points

The local path is not a hash-vector demo. Ollama serves a dedicated neural
embedding model for FAISS and Qwen2 0.5B for generation. Qwen is deliberately a
small baseline; embedding and generation providers are independent so Gemini or
a stronger local model can be evaluated without rewriting retrieval.

LangGraph is justified by control flow: evidence is graded, weak evidence
triggers one bounded query rewrite, figure analysis is conditional, citation
coverage gets one bounded regeneration attempt, and unresolved cases abstain.
The graph is compiled once per service instead of once per question.

Docling is primary for layout, tables, reading order and figures. MarkItDown is
the fast fallback; PyMuPDF is the final PDF fallback. Parser provenance and
fallback reasons remain attached to evidence.

Prometheus is operational observability, not answer quality. Corpus statistics
live on `/stats`; `/metrics` exposes scrape-compatible time-series metrics.

Do not claim improved accuracy until the corpus evaluation and local-versus-
Gemini ablation report has been run. Say “implemented and tested” for code paths
and “to be evaluated” for model-quality comparisons.

