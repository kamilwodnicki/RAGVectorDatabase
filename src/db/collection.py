from qdrant_client.models import Distance, VectorParams, CollectionInfo
from src.db.client import get_client
from src.config import COLLECTION_NAME, EMBEDDING_DIM


def setup_collection(recreate: bool = False) -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
        else:
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def get_collection_info() -> CollectionInfo:
    return get_client().get_collection(COLLECTION_NAME)
