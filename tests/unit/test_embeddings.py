import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def embeddings(monkeypatch):
    """Create an E5HuggingFaceEmbeddings without actually loading the model."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from src.ingest import embeddings as embeddings_module

    monkeypatch.setattr(HuggingFaceEmbeddings, "__init__", lambda self, **kwargs: None)
    inst = embeddings_module.E5HuggingFaceEmbeddings(device="cpu")
    return inst


def test_embed_documents_prepends_passage_prefix(embeddings, monkeypatch):
    captured: list[list[str]] = []

    def fake_super_embed_documents(self, texts):
        captured.append(list(texts))
        return [[0.0] * 3 for _ in texts]

    from langchain_community.embeddings import HuggingFaceEmbeddings
    monkeypatch.setattr(HuggingFaceEmbeddings, "embed_documents", fake_super_embed_documents)

    embeddings.embed_documents(["pierwszy", "drugi"])

    assert captured == [["passage: pierwszy", "passage: drugi"]]


def test_embed_query_prepends_query_prefix(embeddings, monkeypatch):
    captured: list[str] = []

    def fake_super_embed_query(self, text):
        captured.append(text)
        return [0.0] * 3

    from langchain_community.embeddings import HuggingFaceEmbeddings
    monkeypatch.setattr(HuggingFaceEmbeddings, "embed_query", fake_super_embed_query)

    embeddings.embed_query("pytanie testowe")

    assert captured == ["query: pytanie testowe"]


def test_embed_documents_handles_empty_list(embeddings, monkeypatch):
    captured: list[list[str]] = []

    def fake_super_embed_documents(self, texts):
        captured.append(list(texts))
        return []

    from langchain_community.embeddings import HuggingFaceEmbeddings
    monkeypatch.setattr(HuggingFaceEmbeddings, "embed_documents", fake_super_embed_documents)

    embeddings.embed_documents([])

    assert captured == [[]]


def test_embed_query_prefix_prepended_exactly_once(embeddings, monkeypatch):
    captured: list[str] = []

    def fake_super_embed_query(self, text):
        captured.append(text)
        return [0.0]

    from langchain_community.embeddings import HuggingFaceEmbeddings
    monkeypatch.setattr(HuggingFaceEmbeddings, "embed_query", fake_super_embed_query)

    embeddings.embed_query("query: sabotaż")

    assert captured == ["query: query: sabotaż"]
