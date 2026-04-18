from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, API_DEVICE
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
        results = get_vectorstore().similarity_search(request.query, k=request.k)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return QueryResponse(
        results=[DocumentFragment(content=d.page_content, metadata=d.metadata) for d in results]
    )
