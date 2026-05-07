from qdrant_client.models import (
    CollectionInfo,
    Distance,
    PayloadSchemaType,
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


_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,
    "filename": PayloadSchemaType.KEYWORD,
    "file_extension": PayloadSchemaType.KEYWORD,
    "parent_id": PayloadSchemaType.KEYWORD,
    "page": PayloadSchemaType.INTEGER,
    "ingested_at": PayloadSchemaType.DATETIME,
    "article_id": PayloadSchemaType.KEYWORD,
    "article_date": PayloadSchemaType.DATETIME,
    "article_title": PayloadSchemaType.TEXT,
}


def _schema_is_hybrid(info: CollectionInfo) -> bool:
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors:
        return False
    sparse = info.config.params.sparse_vectors or {}
    return SPARSE_VECTOR_NAME in sparse


def _ensure_payload_indexes() -> None:
    client = get_client()
    for field_name, schema in _PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception:
            # Index already exists — Qdrant rzuca wyjątek przy duplikacie. Idempotentne.
            pass


def setup_collection(recreate: bool = False) -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
        else:
            info = client.get_collection(COLLECTION_NAME)
            if _schema_is_hybrid(info):
                _ensure_payload_indexes()
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
    _ensure_payload_indexes()


def get_collection_info() -> CollectionInfo:
    return get_client().get_collection(COLLECTION_NAME)
