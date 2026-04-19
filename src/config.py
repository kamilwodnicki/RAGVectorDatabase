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

PARENT_MAX_SIZE = 2000
PARENT_SOFT_SIZE = 1500
PARENT_COMBINE_UNDER = 800
PARENT_OVERLAP = 0

CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 80

EXTRACTION_STRATEGY = os.getenv("EXTRACTION_STRATEGY", "fast")
EXTRACTION_LANGUAGES = [lang.strip() for lang in os.getenv("EXTRACTION_LANGUAGES", "pol,eng").split(",") if lang.strip()]
