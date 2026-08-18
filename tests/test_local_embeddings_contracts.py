from biorag.local_embeddings import LocalHashEmbeddings
from biorag.models import Document


def test_local_vectors_are_deterministic():
    embeddings = LocalHashEmbeddings(dimensions=16)
    assert embeddings.embed_query("B7-H3 response") == embeddings.embed_query("B7-H3 response")


def test_local_vectors_have_requested_dimensions():
    vector = LocalHashEmbeddings(dimensions=11).embed_query("marker")
    assert len(vector) == 11


def test_local_vectors_are_case_insensitive():
    embeddings = LocalHashEmbeddings(dimensions=16)
    assert embeddings.embed_query("BIOMARKER") == embeddings.embed_query("biomarker")


def test_document_batch_preserves_input_order():
    embeddings = LocalHashEmbeddings(dimensions=32)
    batch = embeddings.embed_documents(["oncology", "cardiology"])
    assert batch == [embeddings.embed_query("oncology"), embeddings.embed_query("cardiology")]


def test_evidence_embedding_combines_title_and_text():
    embeddings = LocalHashEmbeddings(dimensions=32)
    document = Document("1", "Oncology", "B7-H3 response", "paper.pdf")
    assert embeddings.embed_evidence([document]) == [
        embeddings.embed_query("Oncology\nB7-H3 response")
    ]


def test_empty_text_returns_zero_vector():
    assert LocalHashEmbeddings(dimensions=4).embed_query("") == [0.0, 0.0, 0.0, 0.0]

