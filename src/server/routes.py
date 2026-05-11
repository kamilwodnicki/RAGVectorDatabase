import json
import logging
import os
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, HTTPException

from src.config import (
    API_DEVICE,
    BASE_TAG,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    EMBEDDING_DIM,
    EXTRACTION_STRATEGY,
    HYBRID_DENSE_WEIGHT,
    HYBRID_RRF_K,
    HYBRID_SPARSE_WEIGHT,
    MODEL_NAME,
    PARENT_MAX_SIZE,
    PDF_SOURCE_DIR,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_RETRIEVE_K,
    RETRIEVAL_MODE,
    SPARSE_MODEL_NAME,
)
from src.db.metadata_store import MetadataStoreError
from src.db.mongo import get_parents_collection
from src.ingest.embeddings import E5HuggingFaceEmbeddings
from src.ingest.pipeline import (
    SyncResult,
    run_rebuild,
    run_sync,
    run_sync_paths,
)
from src.ingest.sparse_embeddings import BM25SparseEmbeddings
from src.retrieval.filters import InvalidFilterError, build_qdrant_filter
from src.retrieval.hybrid import retrieve_children
from src.retrieval.reranker import RerankCandidate, RerankResult, Reranker
from src.server.metrics import (
    mongo_query_duration_seconds,
    observe,
    qdrant_children_per_query,
    qdrant_query_duration_seconds,
    qdrant_top_score,
    query_parents_returned,
    reranker_candidates_count,
    reranker_duration_seconds,
)

QUERY_LOG = logging.getLogger("rag.query")
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID", "")
from src.server.schemas import (
    DocumentFragment,
    IngestRebuildRequest,
    IngestSyncRequest,
    IngestSyncResponse,
    QueryRequest,
    QueryResponse,
    SyncErrorItem,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_dense_embedder = E5HuggingFaceEmbeddings(device=API_DEVICE)
_sparse_embedder: BM25SparseEmbeddings | None = None
_reranker: Reranker | None = None


def _get_sparse_embedder() -> BM25SparseEmbeddings | None:
    global _sparse_embedder
    if RETRIEVAL_MODE in ("sparse", "hybrid") and _sparse_embedder is None:
        _sparse_embedder = BM25SparseEmbeddings()
    return _sparse_embedder


def _get_reranker() -> Reranker | None:
    global _reranker
    if not RERANKER_ENABLED:
        return None
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


@router.post("/query/", response_model=QueryResponse)
def query(request: QueryRequest):
    started = perf_counter()
    logger.info(
        "Query: k=%d mode=%s reranker=%s filters=%s query=%r",
        request.k, RETRIEVAL_MODE, RERANKER_ENABLED, request.filters, request.query[:200],
    )
    try:
        qdrant_filter = build_qdrant_filter(request.filters)
    except InvalidFilterError as e:
        logger.warning("Filtr odrzucony: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    # Z rerankerem pobieramy więcej kandydatów, żeby reranker miał z czego wybierać.
    retrieve_k = max(RERANKER_RETRIEVE_K, request.k) if RERANKER_ENABLED else request.k

    try:
        with observe(qdrant_query_duration_seconds, operation="retrieve_children"):
            children = retrieve_children(
                query=request.query,
                k=retrieve_k,
                mode=RETRIEVAL_MODE,
                dense_embedder=_dense_embedder,
                sparse_embedder=_get_sparse_embedder(),
                query_filter=qdrant_filter,
            )
    except Exception as e:
        logger.error("Retrieval failed: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

    qdrant_children_per_query.labels(mode=RETRIEVAL_MODE).observe(len(children))
    if children:
        qdrant_top_score.labels(mode=RETRIEVAL_MODE).observe(children[0].score)

    # Dedupe parent_id, zachowując kolejność z RRF, zapamiętując best retrieval score per parent
    retrieval_score_by_parent: dict[str, float] = {}
    ordered_parent_ids: list[str] = []
    for child in children:
        if child.parent_id and child.parent_id not in retrieval_score_by_parent:
            retrieval_score_by_parent[child.parent_id] = child.score
            ordered_parent_ids.append(child.parent_id)

    query_parents_returned.labels(mode=RETRIEVAL_MODE).observe(len(ordered_parent_ids))

    if not ordered_parent_ids:
        _log_query(request, children, [], [], started)
        return QueryResponse(results=[])

    # Fetch wszystkich kandydatów z Mongo — potrzebny pełen tekst do rerankera albo do response'a
    parents_col = get_parents_collection()
    with observe(mongo_query_duration_seconds, operation="find_parents"):
        parent_docs = {
            p["_id"]: p
            for p in parents_col.find({"_id": {"$in": ordered_parent_ids}})
        }

    # Reranking — jeśli włączony, przeszereguj kandydatów po cross-encoder score'ie
    rerank_results: list[RerankResult] = []
    reranker = _get_reranker()
    if reranker is not None:
        candidates = [
            RerankCandidate(
                parent_id=pid,
                text=parent_docs[pid]["text"],
                retrieval_score=retrieval_score_by_parent[pid],
            )
            for pid in ordered_parent_ids
            if pid in parent_docs
        ]
        reranker_candidates_count.observe(len(candidates))
        try:
            with observe(reranker_duration_seconds):
                rerank_results = reranker.rerank(request.query, candidates)
            final_parent_ids = [r.parent_id for r in rerank_results[:request.k]]
        except Exception as e:
            logger.error("Reranker failed, fallback do kolejności retrievalu: %s", e, exc_info=True)
            final_parent_ids = ordered_parent_ids[:request.k]
    else:
        final_parent_ids = ordered_parent_ids[:request.k]

    fragments = []
    for pid in final_parent_ids:
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
                "article_id": p.get("article_id"),
                "article_date": p.get("article_date"),
                "article_title": p.get("article_title"),
            },
        ))

    _log_query(
        request,
        children,
        [f.metadata.get("parent_id") for f in fragments],
        rerank_results,
        started,
    )
    return QueryResponse(results=fragments)


def _log_query(request, children, returned_parent_ids, rerank_results, started):
    """Jeden JSON-owy log line per /query/. Konsumowany przez Loki w Grafanie."""
    record = {
        "event": "query",
        "experiment_id": EXPERIMENT_ID,
        "base_tag": BASE_TAG,
        "query": request.query[:200],
        "k": request.k,
        "mode": RETRIEVAL_MODE,
        "filters": request.filters,
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "child_chunk_size": CHILD_CHUNK_SIZE,
        "child_chunk_overlap": CHILD_CHUNK_OVERLAP,
        "parent_max_size": PARENT_MAX_SIZE,
        "extraction_strategy": EXTRACTION_STRATEGY,
        "n_children": len(children),
        "n_parents": len(returned_parent_ids),
        "top_score": children[0].score if children else None,
        "children": [
            {
                "id": c.id,
                "parent_id": c.parent_id,
                "score": round(c.score, 4),
                "source": c.source,
                "page": c.page,
            }
            for c in children
        ],
        "returned_parent_ids": returned_parent_ids,
        "duration_ms": round((perf_counter() - started) * 1000, 2),
    }
    if RETRIEVAL_MODE == "hybrid":
        record["hybrid"] = {
            "dense_weight": HYBRID_DENSE_WEIGHT,
            "sparse_weight": HYBRID_SPARSE_WEIGHT,
            "rrf_k": HYBRID_RRF_K,
            "sparse_model": SPARSE_MODEL_NAME,
        }
    if RERANKER_ENABLED:
        record["reranker"] = {
            "enabled": True,
            "model": RERANKER_MODEL,
            "retrieve_k": RERANKER_RETRIEVE_K,
            "n_reranked": len(rerank_results),
            "scores": [
                {
                    "parent_id": r.parent_id,
                    "rerank_score": round(r.rerank_score, 4),
                    "retrieval_score": round(r.retrieval_score, 4),
                }
                for r in rerank_results
            ],
        }
    QUERY_LOG.info(json.dumps(record, ensure_ascii=False, default=str))


def _validate_paths_under_source(paths: list[str]) -> list[str]:
    """Defense-in-depth: każda ścieżka musi być wewnątrz PDF_SOURCE_DIR.
    Zwraca znormalizowane ścieżki (resolved). Rzuca HTTPException(400) na pierwszą złą.
    """
    source_root = Path(PDF_SOURCE_DIR).resolve()
    resolved = []
    for raw in paths:
        try:
            r = Path(raw).resolve()
            r.relative_to(source_root)
            resolved.append(str(r))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Ścieżka '{raw}' musi być wewnątrz katalogu '{PDF_SOURCE_DIR}'",
            )
    return resolved


def _sync_result_to_response(result: SyncResult, strategy: str) -> IngestSyncResponse:
    return IngestSyncResponse(
        added=result.added,
        updated=result.updated,
        skipped=result.skipped,
        deleted=result.deleted,
        errors=[SyncErrorItem(path=e.path, error=e.error) for e in result.errors],
        elapsed_seconds=result.elapsed_seconds,
        strategy=strategy,
    )


@router.post("/ingest/sync/", response_model=IngestSyncResponse)
def ingest_sync(request: IngestSyncRequest):
    """Synchronizacja korpusu.

    - Brak `paths` (lub pusta lista) → pełen incremental sync całego DOKUMENTY/
      (akcje ADD/UPDATE/SKIP/DELETE jak w `python manage.py ingest run`).
    - Z `paths` → reprocess **tylko podanych plików**, każdy wymuszony niezależnie
      od hash (jak `python manage.py ingest file <path>`). Bez SKIP/DELETE.

    UWAGA: synchroniczny. Może trwać minuty/godziny dla dużych korpusów lub `hi_res`.
    """
    strategy = request.strategy or EXTRACTION_STRATEGY
    logger.info("Sync request: paths=%s strategy=%s", request.paths, strategy)

    try:
        if request.paths:
            validated = _validate_paths_under_source(request.paths)
            result = run_sync_paths(validated, strategy=strategy)
        else:
            result = run_sync(source_dir=PDF_SOURCE_DIR, strategy=strategy)
    except RuntimeError as e:
        # Brak GPU
        raise HTTPException(status_code=503, detail=str(e))
    except MetadataStoreError as e:
        logger.error("Metadata store error: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return _sync_result_to_response(result, strategy)


@router.post("/ingest/rebuild/", response_model=IngestSyncResponse)
def ingest_rebuild(request: IngestRebuildRequest):
    """Pełna przebudowa — kasuje wszystko z Qdrant + Mongo, potem reindex od zera.

    Wymaga `confirm: "DELETE_ALL"` w body — defense-in-depth, żeby przypadkowy POST
    nie wyczyścił produkcji.
    """
    if request.confirm != "DELETE_ALL":
        raise HTTPException(
            status_code=400,
            detail='Wymagane potwierdzenie: pole "confirm" musi mieć wartość "DELETE_ALL"',
        )

    strategy = request.strategy or EXTRACTION_STRATEGY
    logger.warning("Rebuild request potwierdzony, strategy=%s", strategy)

    try:
        result = run_rebuild(source_dir=PDF_SOURCE_DIR, strategy=strategy)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except MetadataStoreError as e:
        logger.error("Metadata store error: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Rebuild failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return _sync_result_to_response(result, strategy)
