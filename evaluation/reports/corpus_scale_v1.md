# BioRAG corpus scale report v1

Date: 2026-09-13

| Measure | Verified value |
|---|---:|
| Open-access PMC PDFs | 50 |
| PDF pages | 1,046 |
| Extracted text characters | 5,157,073 |
| Page-preserving semantic chunks | 7,734 |

The counts were generated locally from the exact files identified by
`data/corpus/corpus.lock.json`. Every lock entry records its PMCID, filename, byte
size, SHA-256 digest, license label, and evaluation split. PDFs are intentionally
excluded from Git; the manifest and lock make the corpus independently auditable and
re-downloadable.

Bulk counts use PyMuPDF native-text extraction followed by BioRAG's sentence-aware
900-character chunker with one-sentence overlap. Docling remains the layout-aware
path for tables, figures, and OCR-sensitive documents rather than the bulk counting
path. Consequently, these figures describe the reproducible retrieval corpus and do
not claim that all 50 papers received full Docling processing.
