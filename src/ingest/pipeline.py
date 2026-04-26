import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import typer
from qdrant_client.models import PointIdsList, PointStruct, SparseVector

logger = logging.getLogger(__name__)

from src.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    EXTRACTION_STRATEGY,
    INGEST_DEVICE,
    MONGODB_DB,
    MONGODB_FILES_METADATA_COLLECTION,
    PDF_SOURCE_DIR,
    QDRANT_UPSERT_BATCH_SIZE,
    SPARSE_VECTOR_NAME,
    format_effective_config,
)
from src.db.client import get_client
from src.db.collection import setup_collection
from src.db.metadata_store import (
    FileAction,
    MetadataStoreError,
    _compute_file_hash,
    delete_file_metadata,
    evaluate_file_status,
    find_deleted_files,
    mark_file_error,
    upsert_file_metadata,
)
from src.db.mongo import get_mongo_client, get_parents_collection
from src.extractor.pipeline import extract_single_file
from src.ingest.chunker import chunk_file_elements
from src.ingest.embeddings import E5HuggingFaceEmbeddings
from src.ingest.sparse_embeddings import BM25SparseEmbeddings


SUPPORTED_EXTENSIONS = (".pdf", ".txt")


@dataclass
class SingleFileResult:
    parent_ids: list[str]
    child_ids: list[str]
    replaced_existing: bool


def _ensure_cuda() -> None:
    if INGEST_DEVICE is None:
        typer.secho("BŁĄD: Brak GPU (CUDA). Przerwanie operacji.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _list_source_files(source_dir: str) -> list[Path]:
    return [
        p for p in sorted(Path(source_dir).rglob("*"))
        if p.is_file() and p.suffix.lower() in (".pdf", ".txt")
    ]


def _delete_file_vectors(parent_ids: list[str], child_ids: list[str]) -> None:
    if parent_ids:
        get_parents_collection().delete_many({"_id": {"$in": parent_ids}})
    if child_ids:
        get_client().delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=child_ids),
        )


def _ingest_one_file(
    path: Path,
    strategy: str,
    content_hash: str,
    dense_embedder: E5HuggingFaceEmbeddings,
    sparse_embedder: BM25SparseEmbeddings,
) -> tuple[list[str], list[str]]:
    elements = extract_single_file(path, strategy=strategy)
    if not elements:
        return [], []

    parents, children = chunk_file_elements(path, elements)
    if parents:
        get_parents_collection().insert_many(parents)

    parent_ids = [p["_id"] for p in parents]
    child_ids: list[str] = []

    if children:
        texts = [c.page_content for c in children]
        dense_vectors = dense_embedder.embed_documents(texts)
        sparse_vectors = sparse_embedder.embed_documents(texts)

        points = []
        for child, dense_vec, sparse_vec in zip(children, dense_vectors, sparse_vectors):
            child_id = str(uuid.uuid4())
            child_ids.append(child_id)
            points.append(PointStruct(
                id=child_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                },
                payload={
                    "text": child.page_content,
                    "parent_id": child.metadata["parent_id"],
                    "source": child.metadata["source"],
                    "filename": child.metadata["filename"],
                    "file_extension": child.metadata["file_extension"],
                    "page": child.metadata.get("page"),
                    "ingested_at": child.metadata["ingested_at"],
                },
            ))

        try:
            for i in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
                get_client().upsert(
                    collection_name=COLLECTION_NAME,
                    points=points[i:i + QDRANT_UPSERT_BATCH_SIZE],
                )
        except Exception:
            # Rollback — usuń punkty które już zdążyły wejść, żeby nie zostawić orphans
            try:
                get_client().delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=PointIdsList(points=child_ids),
                )
            except Exception:
                pass
            raise

    upsert_file_metadata(
        file_path=str(path),
        content_hash=content_hash,
        parent_doc_ids=parent_ids,
        child_vector_ids=child_ids,
    )
    return parent_ids, child_ids


def run_sync(source_dir: str = PDF_SOURCE_DIR, strategy: str = EXTRACTION_STRATEGY) -> None:
    _ensure_cuda()
    typer.echo(format_effective_config())
    typer.echo(f"Synchronizacja | Urządzenie: {INGEST_DEVICE} | Strategia: {strategy}")

    physical = _list_source_files(source_dir)
    physical_strs = [str(p) for p in physical]

    try:
        evaluations = [evaluate_file_status(str(p)) for p in physical]
        deletions = find_deleted_files(physical_strs)
    except MetadataStoreError as e:
        logger.error("Metadata-store nieosiągalny: %s", e, exc_info=True)
        typer.secho(f"BŁĄD metadata-store: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    logger.info(
        "Start synchronizacji: source=%s strategy=%s nowych=%d zmienionych=%d usuniętych=%d bez_zmian=%d",
        source_dir, strategy,
        sum(1 for e in evaluations if e.action == FileAction.ADD),
        sum(1 for e in evaluations if e.action == FileAction.UPDATE),
        len(deletions),
        sum(1 for e in evaluations if e.action == FileAction.SKIP),
    )

    to_add = [e for e in evaluations if e.action == FileAction.ADD]
    to_update = [e for e in evaluations if e.action == FileAction.UPDATE]
    to_skip = [e for e in evaluations if e.action == FileAction.SKIP]

    typer.echo(
        f"Plan: {len(to_add)} nowych | {len(to_update)} zmienionych | "
        f"{len(deletions)} usuniętych | {len(to_skip)} bez zmian"
    )

    if not (to_add or to_update or deletions):
        typer.secho("Baza jest aktualna — nic do zrobienia.", fg=typer.colors.GREEN)
        return

    setup_collection(recreate=False)

    for ev in deletions:
        try:
            _delete_file_vectors(ev.old_parent_doc_ids, ev.old_child_vector_ids)
            delete_file_metadata(ev.file_path)
            logger.info("DEL %s", ev.file_path)
            typer.echo(f"  [DEL] {ev.file_path}")
        except Exception as e:
            logger.error("ERR-DEL %s: %s", ev.file_path, e, exc_info=True)
            typer.secho(f"  [ERR-DEL] {ev.file_path}: {e}", fg=typer.colors.RED)

    for ev in to_update:
        try:
            _delete_file_vectors(ev.old_parent_doc_ids, ev.old_child_vector_ids)
        except Exception as e:
            logger.error("ERR-UPD-CLEAN %s: %s", ev.file_path, e, exc_info=True)
            typer.secho(f"  [ERR-UPD-CLEAN] {ev.file_path}: {e}", fg=typer.colors.RED)

    to_process = to_add + to_update
    if to_process:
        dense_embedder = E5HuggingFaceEmbeddings(device=INGEST_DEVICE)
        sparse_embedder = BM25SparseEmbeddings()

        for ev in to_process:
            path = Path(ev.file_path)
            label = "ADD" if ev.action == FileAction.ADD else "UPD"
            try:
                parent_ids, child_ids = _ingest_one_file(
                    path, strategy, ev.content_hash, dense_embedder, sparse_embedder
                )
                if not parent_ids and not child_ids:
                    logger.warning("WARN %s: brak wyekstrahowanych elementów", ev.file_path)
                    typer.secho(f"  [WARN] {ev.file_path}: brak wyekstrahowanych elementów", fg=typer.colors.YELLOW)
                    continue
                logger.info(
                    "%s %s (%d parents / %d children)",
                    label, ev.file_path, len(parent_ids), len(child_ids),
                )
                typer.echo(
                    f"  [{label}] {ev.file_path} ({len(parent_ids)}p / {len(child_ids)}c)"
                )
            except Exception as e:
                try:
                    mark_file_error(ev.file_path, ev.content_hash)
                except MetadataStoreError:
                    pass
                logger.error("ERR-%s %s: %s", label, ev.file_path, e, exc_info=True)
                typer.secho(f"  [ERR-{label}] {ev.file_path}: {e}", fg=typer.colors.RED)

    logger.info("Synchronizacja zakończona.")

    typer.secho("Synchronizacja zakończona.", fg=typer.colors.GREEN)


def run_rebuild(source_dir: str = PDF_SOURCE_DIR, strategy: str = EXTRACTION_STRATEGY) -> None:
    _ensure_cuda()
    typer.echo("Czyszczenie baz...")

    setup_collection(recreate=True)
    get_parents_collection().delete_many({})
    get_mongo_client()[MONGODB_DB][MONGODB_FILES_METADATA_COLLECTION].delete_many({})

    typer.echo("  Qdrant: kolekcja utworzona na nowo")
    typer.echo(f"  MongoDB: wyczyszczono '{MONGODB_DB}.parents' i '{MONGODB_DB}.{MONGODB_FILES_METADATA_COLLECTION}'")

    run_sync(source_dir=source_dir, strategy=strategy)


def run_single_file(
    file_path: str,
    strategy: str = EXTRACTION_STRATEGY,
) -> SingleFileResult:
    """Wymusza reprocess jednego pliku — niezależnie od hash w metadata store.

    Stary stan (jeśli istnieje w metadata store) zostaje wyczyszczony z Qdranta i Mongo,
    plik jest re-ekstrahowany, re-embedowany i zapisany od zera.

    Raises:
        FileNotFoundError: ścieżka nie istnieje lub nie jest plikiem.
        ValueError: nieobsługiwane rozszerzenie (dozwolone: .pdf, .txt).
        RuntimeError: brak GPU (CUDA niedostępne).
        MetadataStoreError: błąd komunikacji z Mongo.
    """
    if INGEST_DEVICE is None:
        raise RuntimeError("Ingest wymaga GPU (CUDA), ale jest niedostępne.")

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Plik nie istnieje lub nie jest plikiem: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Nieobsługiwane rozszerzenie '{path.suffix}'. "
            f"Dozwolone: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    setup_collection(recreate=False)

    metadata_col = get_mongo_client()[MONGODB_DB][MONGODB_FILES_METADATA_COLLECTION]
    existing = metadata_col.find_one({"file_path": str(path)})
    replaced = bool(existing)

    if existing:
        old_parent_ids = existing.get("parent_doc_ids", [])
        old_child_ids = existing.get("child_vector_ids", [])
        if old_parent_ids or old_child_ids:
            try:
                _delete_file_vectors(old_parent_ids, old_child_ids)
                logger.info(
                    "FILE-CLEAN %s: usunięto %d parents / %d children",
                    path, len(old_parent_ids), len(old_child_ids),
                )
            except Exception as e:
                logger.error("FILE-ERR-CLEAN %s: %s", path, e, exc_info=True)
                raise

    content_hash = _compute_file_hash(str(path))
    dense_embedder = E5HuggingFaceEmbeddings(device=INGEST_DEVICE)
    sparse_embedder = BM25SparseEmbeddings()

    try:
        parent_ids, child_ids = _ingest_one_file(
            path, strategy, content_hash, dense_embedder, sparse_embedder,
        )
    except Exception as e:
        try:
            mark_file_error(str(path), content_hash)
        except MetadataStoreError:
            pass
        logger.error("FILE-ERR %s: %s", path, e, exc_info=True)
        raise

    logger.info(
        "FILE-DONE %s strategy=%s parents=%d children=%d replaced=%s",
        path, strategy, len(parent_ids), len(child_ids), replaced,
    )

    return SingleFileResult(
        parent_ids=parent_ids,
        child_ids=child_ids,
        replaced_existing=replaced,
    )
