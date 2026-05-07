import os
import torch

PDF_SOURCE_DIR = "./DOKUMENTY"

# BASE_TAG — pojedynczy switch dla wariantu bazy. Każdy tag ma własną kolekcję
# w Qdrant ('documents_<tag>') i własną bazę w MongoDB ('rag_<tag>'), więc
# zmiana modelu/wymiarów wektora/chunkingu tworzy NOWĄ bazę bez nadpisywania
# poprzedniej. Switch między wariantami = edycja .env + restart rag-server.
BASE_TAG = os.getenv("BASE_TAG", "default")

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Styl promptu embedderów. Różne rodziny modeli potrzebują różnego prompting:
#   e5    — prefix "passage: " (dokumenty) i "query: " (zapytanie). Dla rodziny intfloat/e5*.
#   qwen3 — Qwen3-Embedding: dokumenty raw, zapytanie z instrukcją.
#   none  — bez prefixu. Dla modeli typu BGE-M3, MMLW i innych "raw input" embedderów.
EMBEDDING_PROMPT_STYLE = os.getenv("EMBEDDING_PROMPT_STYLE", "e5").lower()

# Instrukcja dla Qwen3 (i kompatybilnych instruction-tuned embedderów). Domyślna
# dobra dla retrieval; możesz przestawić per-domena jeśli eksperymentujesz.
QWEN3_INSTRUCTION = os.getenv(
    "QWEN3_INSTRUCTION",
    "Given a web search query, retrieve relevant passages that answer the query",
)

API_DEVICE = "cpu"
INGEST_DEVICE = "cuda" if torch.cuda.is_available() else None

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
# Override przez COLLECTION_NAME (rzadkie — np. dla testów); domyślnie
# wynika z BASE_TAG.
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or f"documents_{BASE_TAG}"

MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
# Override przez MONGODB_DB; domyślnie wynika z BASE_TAG.
MONGODB_DB = os.getenv("MONGODB_DB") or f"rag_{BASE_TAG}"
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
        f"  Base tag:        {BASE_TAG}",
        f"  Qdrant:          host={QDRANT_HOST} port={QDRANT_PORT} collection={COLLECTION_NAME}",
        f"  MongoDB:         host={MONGODB_HOST} port={MONGODB_PORT} db={MONGODB_DB}",
        f"  Model:           {MODEL_NAME} (dim={EMBEDDING_DIM} style={EMBEDDING_PROMPT_STYLE})",
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
