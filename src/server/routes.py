import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.config import API_DEVICE, EXTRACTION_STRATEGY, PDF_SOURCE_DIR, RETRIEVAL_MODE
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
from src.server.metrics import (
    mongo_query_duration_seconds,
    observe,
    qdrant_query_duration_seconds,
)
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
        with observe(qdrant_query_duration_seconds, operation="retrieve_children"):
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
    with observe(mongo_query_duration_seconds, operation="find_parents"):
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
