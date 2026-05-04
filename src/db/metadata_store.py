import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from src.config import MONGODB_DB, MONGODB_FILES_METADATA_COLLECTION
from src.db.mongo import get_mongo_client


class FileAction(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    SKIP = "SKIP"
    DELETE = "DELETE"


class FileStatus(str, Enum):
    PROCESSED = "PROCESSED"
    ERROR = "ERROR"


@dataclass
class FileEvaluation:
    action: FileAction
    file_path: str
    content_hash: str | None = None
    old_parent_doc_ids: list[str] = field(default_factory=list)
    old_child_vector_ids: list[str] = field(default_factory=list)


class MetadataStoreError(RuntimeError):
    pass


_HASH_CHUNK_SIZE = 1 << 16
_index_ensured = False


def _compute_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
    except OSError as e:
        raise MetadataStoreError(f"Nie można odczytać pliku '{file_path}': {e}") from e
    return h.hexdigest()


def _get_collection() -> Collection:
    global _index_ensured
    try:
        col = get_mongo_client()[MONGODB_DB][MONGODB_FILES_METADATA_COLLECTION]
        if not _index_ensured:
            col.create_index("file_path", unique=True)
            _index_ensured = True
        return col
    except PyMongoError as e:
        raise MetadataStoreError(f"Błąd połączenia z MongoDB: {e}") from e


def evaluate_file_status(file_path: str) -> FileEvaluation:
    content_hash = _compute_file_hash(file_path)
    col = _get_collection()

    try:
        record = col.find_one({"file_path": file_path})
    except PyMongoError as e:
        raise MetadataStoreError(
            f"Zapytanie do MongoDB nie powiodło się dla '{file_path}': {e}"
        ) from e

    if record is None:
        return FileEvaluation(
            action=FileAction.ADD,
            file_path=file_path,
            content_hash=content_hash,
        )

    if record.get("content_hash") == content_hash:
        return FileEvaluation(
            action=FileAction.SKIP,
            file_path=file_path,
            content_hash=content_hash,
        )

    return FileEvaluation(
        action=FileAction.UPDATE,
        file_path=file_path,
        content_hash=content_hash,
        old_parent_doc_ids=record.get("parent_doc_ids", []),
        old_child_vector_ids=record.get("child_vector_ids", []),
    )


def find_deleted_files(current_physical_files_list: Iterable[str]) -> list[FileEvaluation]:
    current_set = set(current_physical_files_list)
    col = _get_collection()
    deleted_evaluations = []
    try:
        cursor = col.find({}, {"file_path": 1, "parent_doc_ids": 1, "child_vector_ids": 1})
        for record in cursor:
            db_file_path = record.get("file_path")
            if db_file_path not in current_set:
                deleted_evaluations.append(
                    FileEvaluation(
                        action=FileAction.DELETE,
                        file_path=db_file_path,
                        content_hash="",
                        old_parent_doc_ids=record.get("parent_doc_ids", []),
                        old_child_vector_ids=record.get("child_vector_ids", []),
                    )
                )
        return deleted_evaluations
    except PyMongoError as e:
        raise MetadataStoreError(
            f"Nie udało się pobrać listy usuniętych plików: {e}"
        ) from e


def upsert_file_metadata(
    file_path: str,
    content_hash: str,
    parent_doc_ids: list[str],
    child_vector_ids: list[str],
    status: FileStatus = FileStatus.PROCESSED,
) -> None:
    col = _get_collection()
    try:
        col.update_one(
            {"file_path": file_path},
            {
                "$set": {
                    "file_path": file_path,
                    "content_hash": content_hash,
                    "parent_doc_ids": parent_doc_ids,
                    "child_vector_ids": child_vector_ids,
                    "status": status.value,
                    "last_updated": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
    except PyMongoError as e:
        raise MetadataStoreError(
            f"Nie udało się zapisać metadanych dla '{file_path}': {e}"
        ) from e


def delete_file_metadata(file_path: str) -> None:
    col = _get_collection()
    try:
        col.delete_one({"file_path": file_path})
    except PyMongoError as e:
        raise MetadataStoreError(
            f"Nie udało się usunąć metadanych dla '{file_path}': {e}"
        ) from e


def mark_file_error(file_path: str, content_hash: str | None = None) -> None:
    col = _get_collection()
    update: dict = {
        "status": FileStatus.ERROR.value,
        "last_updated": datetime.now(timezone.utc),
        "file_path": file_path,
    }
    if content_hash is not None:
        update["content_hash"] = content_hash
    try:
        col.update_one(
            {"file_path": file_path},
            {"$set": update},
            upsert=True,
        )
    except PyMongoError as e:
        raise MetadataStoreError(
            f"Nie udało się oznaczyć pliku '{file_path}' jako ERROR: {e}"
        ) from e
