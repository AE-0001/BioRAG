import json
from pathlib import Path

from biorag.service import BioRAGService


def main() -> None:
    service = BioRAGService(Path("data/index/eval-index.json"))
    service.ingest([Path("data/demo/biomarkers.md"), Path("data/demo/trial.figures.json")])
    cases = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))
    retrieval_hits = 0
    answer_terms = 0
    grounding_hits = 0
    for case in cases:
        result = service.ask(case["question"], top_k=3)
        retrieval_hits += any(
            hit.document.source == case["expected_source"] for hit in result.evidence
        )
        answer_terms += all(term.lower() in result.answer.lower() for term in case["expected_terms"])
        grounding_hits += result.grounded
    total = len(cases)
    print(
        json.dumps(
            {
                "cases": total,
                "retrieval_recall_at_3": retrieval_hits / total,
                "answer_term_recall": answer_terms / total,
                "citation_grounding_rate": grounding_hits / total,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

