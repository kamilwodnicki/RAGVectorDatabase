import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def embedder():
    from src.ingest.sparse_embeddings import BM25SparseEmbeddings

    return BM25SparseEmbeddings()


def test_embed_documents_returns_one_vector_per_text(embedder):
    texts = ["pierwszy dokument", "drugi dokument", "trzeci dokument"]
    result = embedder.embed_documents(texts)

    assert len(result) == 3


def test_embed_query_returns_single_sparse_vector(embedder):
    result = embedder.embed_query("jakieś pytanie")

    assert hasattr(result, "indices")
    assert hasattr(result, "values")
    assert len(result.indices) == len(result.values)


def test_embed_documents_produces_nonempty_sparse_vectors(embedder):
    result = embedder.embed_documents(["LoRA to metoda dostrajania modeli"])

    first = result[0]
    assert len(first.indices) > 0
    assert len(first.values) == len(first.indices)
    assert all(v >= 0 for v in first.values)


def test_embed_documents_and_query_have_same_structure(embedder):
    doc_vec = embedder.embed_documents(["tekst dokumentu"])[0]
    query_vec = embedder.embed_query("tekst dokumentu")

    assert type(doc_vec) == type(query_vec)
    assert hasattr(doc_vec, "indices") and hasattr(query_vec, "indices")


def test_indices_are_valid_token_ids(embedder):
    result = embedder.embed_documents(["prosty test"])[0]

    assert all(isinstance(i, int) and i >= 0 for i in result.indices)
