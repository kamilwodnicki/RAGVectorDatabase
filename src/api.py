from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import os
from src.core.config import BASE_DB_DIR, API_DEVICE, SPLITTER_TYPES, LENGTH_VARIANTS
from src.core.engine import E5HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma

app = FastAPI()
embeddings = E5HuggingFaceEmbeddings(device=API_DEVICE)
db_cache = {}

class QueryRequest(BaseModel):
    query: str
    db_type: Optional[str] = None
    db_variant: Optional[str] = None
    k: int = 3

def list_available_databases() -> List[str]:
    available = []
    for t in SPLITTER_TYPES:
        for v in LENGTH_VARIANTS.keys():
            path = os.path.join(BASE_DB_DIR, t, v)
            if os.path.exists(path):
                available.append(f"{t}/{v}")
    return available

@app.post("/query/")
def query_rag(request: QueryRequest):
    if not request.db_type or not request.db_variant:
        return {
            "message": "Nie wybrano bazy danych. Wybierz jedną z dostępnych wersji.",
            "available_databases": list_available_databases()
        }

    path = os.path.join(BASE_DB_DIR, request.db_type, request.db_variant)
    
    if not os.path.exists(path):
        return {
            "error": "Wybrana baza nie istnieje.",
            "available_databases": list_available_databases()
        }

    if path not in db_cache:
        client = chromadb.PersistentClient(path=path)
        db_cache[path] = Chroma(client=client, collection_name="docs", embedding_function=embeddings)
    
    results = db_cache[path].similarity_search(request.query, k=request.k)
    return {
        "db_used": f"{request.db_type}/{request.db_variant}",
        "results": [{"content": d.page_content, "metadata": d.metadata} for d in results]
    }