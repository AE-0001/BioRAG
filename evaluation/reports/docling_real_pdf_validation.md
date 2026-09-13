# Docling real-PDF integration validation

Date: 2026-09-13

## Outcome

Docling successfully completed layout-aware processing for 11 of the 13 CC BY
biomedical PDFs before the deliberately interrupted full-corpus run. The completed
set contains 207 pages and produced 98 persisted figure images. The two remaining
papers were not counted as successes.

| Measure | Observed value |
|---|---:|
| Corpus PDFs available | 13 |
| PDFs completed by Docling | 11 |
| Pages in completed PDFs | 207 |
| Extracted figure images | 98 |
| Approximate elapsed wall time | 12.2 minutes |
| Peak Python working set observed | 3.1 GB |

The run used Docling PDF layout analysis, OCR, table structure recognition and
picture-image generation configured in `DoclingPaperIngestor`. Completion was
verified by matching each artifact directory to the first 16 hexadecimal characters
of its source PDF SHA-256 digest. PDFs and generated images remain untracked; corpus
identity is recorded in `data/corpus/corpus.lock.json`.

## Engineering decision

Full Docling processing is retained for documents that require layout, table or
figure understanding. Bulk text retrieval uses PyMuPDF and preserves page provenance.
This tiered policy avoids paying the roughly 3 GB memory and layout-model cost for
every ordinary born-digital page while preserving a multimodal path when it matters.

## Limitations

- This validates successful real-document execution and resource behavior; it is not
  a labelled table- or figure-extraction accuracy study.
- Wall time is reconstructed from the process start and final completed artifact
  timestamp, so it is reported approximately.
- The run was intentionally stopped after 11 completed PDFs; no result is claimed for
  `PMC9489212.1.pdf` or `PMC9727400.1.pdf`.
