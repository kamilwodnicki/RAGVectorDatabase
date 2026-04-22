from pathlib import Path

import typer
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import PointIdsList

from src.config import (
    COLLECTION_NAME,
    EXTRACTION_STRATEGY,
    INGEST_DEVICE,
    MONGODB_DB,
    MONGODB_FILES_METADATA_COLLECTION,
    PDF_SOURCE_DIR,
    QDRANT_HOST,
    QDRANT_PORT,
    format_effective_config,
)
from src.db.client import get_client
from src.db.collection import setup_collection
from src.db.metadata_store import (
    FileAction,
    MetadataStoreError,
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


def _ensure_cuda() -> None:
    if INGEST_DEVICE is None:
        typer.secho("BŁĄD: Brak GPU (CUDA). Przerwanie operacji.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _list_source_files(source_dir: str) -> list[Path]:
    return [
        p for p in sorted(Path(source_dir).rglob("*"))
        if p.is_file() and p.suffix.lower() in (".pdf", ".txt")
    ]


def _get_vectorstore(embeddings: E5HuggingFaceEmbeddings) -> QdrantVectorStore:
    from qdrant_client import QdrantClient
    return QdrantVectorStore(
        client=QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT),
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


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
    vectorstore: QdrantVectorStore,
) -> tuple[list[str], list[str]]:
    elements = extract_single_file(path, strategy=strategy)
    if not elements:
        return [], []

    parents, children = chunk_file_elements(path, elements)
    if parents:
        get_parents_collection().insert_many(parents)

    parent_ids = [p["_id"] for p in parents]
    child_ids = vectorstore.add_documents(children) if children else []

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
        typer.secho(f"BŁĄD metadata-store: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

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
            typer.echo(f"  [DEL] {ev.file_path}")
        except Exception as e:
            typer.secho(f"  [ERR-DEL] {ev.file_path}: {e}", fg=typer.colors.RED)

    for ev in to_update:
        try:
            _delete_file_vectors(ev.old_parent_doc_ids, ev.old_child_vector_ids)
        except Exception as e:
            typer.secho(f"  [ERR-UPD-CLEAN] {ev.file_path}: {e}", fg=typer.colors.RED)

    to_process = to_add + to_update
    if to_process:
        embeddings = E5HuggingFaceEmbeddings(device=INGEST_DEVICE)
        vectorstore = _get_vectorstore(embeddings)

        for ev in to_process:
            path = Path(ev.file_path)
            label = "ADD" if ev.action == FileAction.ADD else "UPD"
            try:
                parent_ids, child_ids = _ingest_one_file(
                    path, strategy, ev.content_hash, vectorstore
                )
                if not parent_ids and not child_ids:
                    typer.secho(f"  [WARN] {ev.file_path}: brak wyekstrahowanych elementów", fg=typer.colors.YELLOW)
                    continue
                typer.echo(
                    f"  [{label}] {ev.file_path} ({len(parent_ids)}p / {len(child_ids)}c)"
                )
            except Exception as e:
                try:
                    mark_file_error(ev.file_path, ev.content_hash)
                except MetadataStoreError:
                    pass
                typer.secho(f"  [ERR-{label}] {ev.file_path}: {e}", fg=typer.colors.RED)

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
