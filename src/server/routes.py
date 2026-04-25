import logging

from fastapi import APIRouter, HTTPException

from src.config import API_DEVICE, RETRIEVAL_MODE
from src.db.mongo import get_parents_collection
from src.ingest.embeddings import E5HuggingFaceEmbeddings
from src.ingest.sparse_embeddings import BM25SparseEmbeddings
from src.retrieval.filters import InvalidFilterError, build_qdrant_filter
from src.retrieval.hybrid import retrieve_children
from src.server.schemas import DocumentFragment, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_dense_embedder = E5HuggingFaceEmbeddings(device=API_DEVICE)
_sparse_embedder: BM25SparseEmbeddings | None = None


def _get_sparse_embedder() -> BM25SparseEmbeddings | None:
    global _sparse_embedder
    if RETRIEVAL_MODE in ("sparse", "hybrid") and _sparse_embedder is None:
        _sparse_embedder = BM25SparseEmbeddings()
    return _sparse_embedder


@router.post("/query/", response_model=QueryResponse)
def query(request: QueryRequest):
    logger.info(
        "Query: k=%d mode=%s filters=%s query=%r",
        request.k, RETRIEVAL_MODE, request.filters, request.query[:200],
    )
    try:
        qdrant_filter = build_qdrant_filter(request.filters)
    except InvalidFilterError as e:
        logger.warning("Filtr odrzucony: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    try:
        children = retrieve_children(
            query=request.query,
            k=request.k,
            mode=RETRIEVAL_MODE,
            dense_embedder=_dense_embedder,
            sparse_embedder=_get_sparse_embedder(),
            query_filter=qdrant_filter,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

    seen: set[str] = set()
    ordered_parent_ids: list[str] = []
    for child in children:
        if child.parent_id and child.parent_id not in seen:
            seen.add(child.parent_id)
            ordered_parent_ids.append(child.parent_id)

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
            metadata={
                "source": p.get("source"),
                "filename": p.get("filename"),
                "file_extension": p.get("file_extension"),
                "page": p.get("page"),
                "ingested_at": p.get("ingested_at"),
                "parent_id": pid,
            },
        ))

    return QueryResponse(results=fragments)
