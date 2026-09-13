from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

try:
    from prometheus_client import Counter, Gauge, Histogram

    QUERY_TOTAL = Counter("biorag_queries_total", "BioRAG questions", ["provider", "outcome"])
    QUERY_SECONDS = Histogram("biorag_query_duration_seconds", "End-to-end query latency")
    RETRIEVAL_SECONDS = Histogram("biorag_retrieval_duration_seconds", "Retrieval latency")
    INGEST_SECONDS = Histogram("biorag_ingest_duration_seconds", "Ingestion latency")
    PARSER_TOTAL = Counter("biorag_parser_total", "Parser outcomes", ["parser", "outcome"])
    INDEX_CHUNKS = Gauge("biorag_index_chunks", "Currently indexed chunks")
    RETRIEVAL_ATTEMPTS = Histogram(
        "biorag_retrieval_attempts", "Retrieval attempts per question", buckets=(1, 2, 3, 4)
    )
except ImportError:  # pragma: no cover - optional at import time
    QUERY_TOTAL = QUERY_SECONDS = RETRIEVAL_SECONDS = INGEST_SECONDS = None
    PARSER_TOTAL = INDEX_CHUNKS = RETRIEVAL_ATTEMPTS = None


@contextmanager
def observe(histogram):
    started = perf_counter()
    try:
        yield
    finally:
        if histogram is not None:
            histogram.observe(perf_counter() - started)
