import os

os.environ["COLLECTION_NAME"] = "documents_test"
os.environ["MONGODB_DB"] = "rag_test"

import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def _shared_dense_embeddings():
    from src.config import INGEST_DEVICE
    from src.ingest.embeddings import E5HuggingFaceEmbeddings

    if INGEST_DEVICE is None:
        pytest.skip("Integration tests require CUDA (INGEST_DEVICE is None)")
    return E5HuggingFaceEmbeddings(device=INGEST_DEVICE)


@pytest.fixture(scope="session")
def _shared_sparse_embeddings():
    from src.ingest.sparse_embeddings import BM25SparseEmbeddings

    return BM25SparseEmbeddings()


@pytest.fixture(autouse=True)
def _reuse_embeddings(_shared_dense_embeddings, _shared_sparse_embeddings, monkeypatch):
    from src.ingest import pipeline

    monkeypatch.setattr(
        pipeline,
        "E5HuggingFaceEmbeddings",
        lambda device: _shared_dense_embeddings,
    )
    monkeypatch.setattr(
        pipeline,
        "BM25SparseEmbeddings",
        lambda: _shared_sparse_embeddings,
    )


@pytest.fixture(autouse=True)
def clean_stores():
    from src.config import COLLECTION_NAME, MONGODB_DB
    from src.db import metadata_store
    from src.db.client import get_client
    from src.db.mongo import get_mongo_client

    qdrant = get_client()
    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    mongo = get_mongo_client()
    mongo.drop_database(MONGODB_DB)
    metadata_store._index_ensured = False

    yield

    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    mongo.drop_database(MONGODB_DB)
    metadata_store._index_ensured = False


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    return tmp_path / "src"


@pytest.fixture
def copy_fixture(source_dir):
    def _copy(filename: str, dest_name: str | None = None) -> Path:
        source_dir.mkdir(parents=True, exist_ok=True)
        dest = source_dir / (dest_name or filename)
        shutil.copy(FIXTURES_DIR / filename, dest)
        return dest
    return _copy
