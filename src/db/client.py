from qdrant_client import QdrantClient
from src.config import QDRANT_HOST, QDRANT_PORT


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
