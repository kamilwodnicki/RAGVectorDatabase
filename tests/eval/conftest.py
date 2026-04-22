import os

os.environ["COLLECTION_NAME"] = "documents_eval"
os.environ["MONGODB_DB"] = "rag_eval"

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"
GOLDEN_SET_PATH = FIXTURES_DIR / "golden_set.json"


@pytest.fixture(scope="session")
def shared_dense_embeddings():
    from src.config import INGEST_DEVICE
    from src.ingest.embeddings import E5HuggingFaceEmbeddings

    if INGEST_DEVICE is None:
        pytest.skip("Eval tests require CUDA (INGEST_DEVICE is None)")
    return E5HuggingFaceEmbeddings(device=INGEST_DEVICE)


@pytest.fixture(scope="session")
def shared_sparse_embeddings():
    from src.ingest.sparse_embeddings import BM25SparseEmbeddings

    return BM25SparseEmbeddings()


@pytest.fixture(scope="session")
def golden_set():
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture(scope="session")
def ingested_corpus(shared_dense_embeddings, shared_sparse_embeddings):
    from src.config import COLLECTION_NAME, MONGODB_DB
    from src.db import metadata_store
    from src.db.client import get_client
    from src.db.mongo import get_mongo_client
    from src.ingest import pipeline
    from src.ingest.pipeline import run_sync

    qdrant = get_client()
    mongo = get_mongo_client()

    def _wipe():
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        mongo.drop_database(MONGODB_DB)
        metadata_store._index_ensured = False

    _wipe()

    original_dense_cls = pipeline.E5HuggingFaceEmbeddings
    original_sparse_cls = pipeline.BM25SparseEmbeddings
    pipeline.E5HuggingFaceEmbeddings = lambda device: shared_dense_embeddings
    pipeline.BM25SparseEmbeddings = lambda: shared_sparse_embeddings
    try:
        run_sync(source_dir=str(CORPUS_DIR))
    finally:
        pipeline.E5HuggingFaceEmbeddings = original_dense_cls
        pipeline.BM25SparseEmbeddings = original_sparse_cls

    yield

    _wipe()
