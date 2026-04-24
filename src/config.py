import os
import torch

PDF_SOURCE_DIR = "./DOKUMENTY"
MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

API_DEVICE = "cpu"
INGEST_DEVICE = "cuda" if torch.cuda.is_available() else None

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
MONGODB_DB = os.getenv("MONGODB_DB", "rag")
MONGODB_PARENTS_COLLECTION = "parents"
MONGODB_FILES_METADATA_COLLECTION = "files_metadata"

PARENT_MAX_SIZE = int(os.getenv("PARENT_MAX_SIZE", "2000"))
PARENT_SOFT_SIZE = int(os.getenv("PARENT_SOFT_SIZE", "1500"))
PARENT_COMBINE_UNDER = int(os.getenv("PARENT_COMBINE_UNDER", "800"))
PARENT_OVERLAP = int(os.getenv("PARENT_OVERLAP", "0"))

CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "80"))

DEFAULT_K = int(os.getenv("DEFAULT_K", "3"))

QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "256"))

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")

RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "1.0"))
HYBRID_SPARSE_WEIGHT = float(os.getenv("HYBRID_SPARSE_WEIGHT", "1.0"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))

EXTRACTION_STRATEGY = os.getenv("EXTRACTION_STRATEGY", "fast")
EXTRACTION_LANGUAGES = [lang.strip() for lang in os.getenv("EXTRACTION_LANGUAGES", "pol,eng").split(",") if lang.strip()]


def format_effective_config() -> str:
    lines = [
        "Efektywna konfiguracja:",
        f"  Qdrant:          host={QDRANT_HOST} port={QDRANT_PORT} collection={COLLECTION_NAME}",
        f"  MongoDB:         host={MONGODB_HOST} port={MONGODB_PORT} db={MONGODB_DB}",
        f"  Model:           {MODEL_NAME} (dim={EMBEDDING_DIM})",
        f"  Urządzenia:      ingest={INGEST_DEVICE} api={API_DEVICE}",
        f"  Parent chunking: max={PARENT_MAX_SIZE} soft={PARENT_SOFT_SIZE} "
        f"combine_under={PARENT_COMBINE_UNDER} overlap={PARENT_OVERLAP}",
        f"  Child chunking:  size={CHILD_CHUNK_SIZE} overlap={CHILD_CHUNK_OVERLAP}",
        f"  Retrieval:       mode={RETRIEVAL_MODE} default_k={DEFAULT_K}",
        f"  Hybrid:          dense_weight={HYBRID_DENSE_WEIGHT} "
        f"sparse_weight={HYBRID_SPARSE_WEIGHT} rrf_k={HYBRID_RRF_K} "
        f"sparse_model={SPARSE_MODEL_NAME}",
        f"  Ekstrakcja:      strategy={EXTRACTION_STRATEGY} languages={EXTRACTION_LANGUAGES}",
    ]
    return "\n".join(lines)
