from qdrant_client.models import (
    CollectionInfo,
    Distance,
    SparseVectorParams,
    VectorParams,
)

from src.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    EMBEDDING_DIM,
    SPARSE_VECTOR_NAME,
)
from src.db.client import get_client


class CollectionSchemaMismatchError(RuntimeError):
    pass


def _schema_is_hybrid(info: CollectionInfo) -> bool:
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors:
        return False
    sparse = info.config.params.sparse_vectors or {}
    return SPARSE_VECTOR_NAME in sparse


def setup_collection(recreate: bool = False) -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
        else:
            info = client.get_collection(COLLECTION_NAME)
            if _schema_is_hybrid(info):
                return
            raise CollectionSchemaMismatchError(
                f"Kolekcja '{COLLECTION_NAME}' ma stary schemat (brak named vectors "
                f"'{DENSE_VECTOR_NAME}'/'{SPARSE_VECTOR_NAME}'). "
                "Uruchom `python manage.py ingest rebuild`, aby zmigrować na hybrid."
            )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )


def get_collection_info() -> CollectionInfo:
    return get_client().get_collection(COLLECTION_NAME)
