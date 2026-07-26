from biorag.chunking import semantic_chunks


def test_chunking_preserves_all_sentences():
    text = "Alpha is first. Beta is second. Gamma is third."
    chunks = semantic_chunks(text, max_chars=30, overlap_sentences=0)
    assert " ".join(chunks) == text
    assert len(chunks) > 1


def test_empty_text_has_no_chunks():
    assert semantic_chunks("   ") == []

