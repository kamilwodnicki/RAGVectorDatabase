from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, API_DEVICE
from src.db.mongo import get_parents_collection
from src.ingest.embeddings import E5HuggingFaceEmbeddings
from src.server.schemas import QueryRequest, QueryResponse, DocumentFragment

router = APIRouter()

_embeddings = E5HuggingFaceEmbeddings(device=API_DEVICE)
_vectorstore: QdrantVectorStore | None = None


def get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=_embeddings,
        )
    return _vectorstore


@router.post("/query/", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        children = get_vectorstore().similarity_search(request.query, k=request.k)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    seen: set[str] = set()
    ordered_parent_ids: list[str] = []
    for child in children:
        pid = child.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            ordered_parent_ids.append(pid)

    if not ordered_parent_ids:
        return QueryResponse(results=[])

    parents_col = get_parents_collection()
    parent_docs = {
        p["_id"]: p
        for p in parents_col.find({"_id": {"$in": ordered_parent_ids}})
    }

    fragments = []
    for pid in ordered_parent_ids:
        p = parent_docs.get(pid)
        if not p:
            continue
        fragments.append(DocumentFragment(
            content=p["text"],
            metadata={"source": p.get("source"), "page": p.get("page"), "parent_id": pid},
        ))

    return QueryResponse(results=fragments)
