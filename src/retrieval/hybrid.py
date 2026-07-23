from dataclasses import dataclass

from qdrant_client.models import Filter, SparseVector

from src.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    HYBRID_DENSE_WEIGHT,
    HYBRID_RRF_K,
    HYBRID_SPARSE_WEIGHT,
    QDRANT_CLIENT_TIMEOUT,
    SPARSE_VECTOR_NAME,
)
from src.db.client import get_client
from src.ingest.embeddings import E5HuggingFaceEmbeddings
from src.ingest.sparse_embeddings import BM25SparseEmbeddings

VALID_MODES = ("dense", "sparse", "hybrid")


@dataclass
class ChildHit:
    id: str
    text: str
    parent_id: str | None
    source: str | None
    page: int | None
    score: float


def retrieve_children(
    query: str,
    k: int,
    mode: str,
    dense_embedder: E5HuggingFaceEmbeddings,
    sparse_embedder: BM25SparseEmbeddings | None,
    query_filter: Filter | None = None,
) -> list[ChildHit]:
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Nieznany RETRIEVAL_MODE '{mode}', dozwolone: {VALID_MODES}")

    if mode == "dense":
        return _search_dense(query, k, dense_embedder, query_filter)
    if mode == "sparse":
        _require_sparse_embedder(sparse_embedder, mode)
        return _search_sparse(query, k, sparse_embedder, query_filter)

    _require_sparse_embedder(sparse_embedder, mode)
    return _search_hybrid(query, k, dense_embedder, sparse_embedder, query_filter)


def weighted_rrf(
    dense_ranking: list[str],
    sparse_ranking: list[str],
    k: int,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    sparse_weight: float = HYBRID_SPARSE_WEIGHT,
    rrf_k: int = HYBRID_RRF_K,
) -> list[tuple[str, float]]:
    """Zwraca top-k (id, score) posortowane malejąco po score."""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ranking, start=1):
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (rrf_k + rank)
    for rank, cid in enumerate(sparse_ranking, start=1):
        scores[cid] = scores.get(cid, 0.0) + sparse_weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def _search_dense(
    query: str,
    k: int,
    embedder: E5HuggingFaceEmbeddings,
    query_filter: Filter | None = None,
) -> list[ChildHit]:
    vec = embedder.embed_query(query)
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        using=DENSE_VECTOR_NAME,
        limit=k,
        with_payload=True,
        query_filter=query_filter,
        timeout=QDRANT_CLIENT_TIMEOUT,
    )
    return [_hit_to_child(h) for h in response.points]


def _search_sparse(
    query: str,
    k: int,
    embedder: BM25SparseEmbeddings,
    query_filter: Filter | None = None,
) -> list[ChildHit]:
    sp = embedder.embed_query(query)
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=SparseVector(indices=sp.indices, values=sp.values),
        using=SPARSE_VECTOR_NAME,
        limit=k,
        with_payload=True,
        query_filter=query_filter,
        timeout=QDRANT_CLIENT_TIMEOUT,
    )
    return [_hit_to_child(h) for h in response.points]


def _search_hybrid(
    query: str,
    k: int,
    dense_embedder: E5HuggingFaceEmbeddings,
    sparse_embedder: BM25SparseEmbeddings,
    query_filter: Filter | None = None,
) -> list[ChildHit]:
    # Pobieramy więcej kandydatów z każdej ścieżki niż docelowe k,
    # żeby fuzja miała z czego wybierać. Ani zbyt mało (utrata dokumentów
    # które są blisko topu tylko w jednym rankingu), ani zbyt dużo
    # (narzut computational).
    fetch = max(k * 3, 30)
    dense_hits = _search_dense(query, fetch, dense_embedder, query_filter)
    sparse_hits = _search_sparse(query, fetch, sparse_embedder, query_filter)

    docs: dict[str, ChildHit] = {}
    for hit in dense_hits:
        docs[hit.id] = hit
    for hit in sparse_hits:
        docs.setdefault(hit.id, hit)

    ranked = weighted_rrf(
        dense_ranking=[h.id for h in dense_hits],
        sparse_ranking=[h.id for h in sparse_hits],
        k=k,
    )

    results = []
    for cid, fused_score in ranked:
        hit = docs[cid]
        results.append(ChildHit(
            id=hit.id,
            text=hit.text,
            parent_id=hit.parent_id,
            source=hit.source,
            page=hit.page,
            score=fused_score,
        ))
    return results


def _hit_to_child(hit) -> ChildHit:
    payload = hit.payload or {}
    return ChildHit(
        id=str(hit.id),
        text=payload.get("text", ""),
        parent_id=payload.get("parent_id"),
        source=payload.get("source"),
        page=payload.get("page"),
        score=hit.score if hit.score is not None else 0.0,
    )


def _require_sparse_embedder(emb, mode: str) -> None:
    if emb is None:
        raise ValueError(
            f"Tryb '{mode}' wymaga sparse_embedder (np. BM25SparseEmbeddings), "
            "ale przekazano None."
        )
