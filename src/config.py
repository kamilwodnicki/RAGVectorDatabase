import os
import torch

PDF_SOURCE_DIR = "./DOKUMENTY"
MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

API_DEVICE = "cpu"
INGEST_DEVICE = "cuda" if torch.cuda.is_available() else None

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
COLLECTION_NAME = "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 160

EXTRACTION_STRATEGY = os.getenv("EXTRACTION_STRATEGY", "fast")
