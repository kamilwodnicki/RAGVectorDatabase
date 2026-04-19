import hashlib
from pathlib import Path

import mongomock
import pytest

from src.db import metadata_store
from src.db.metadata_store import (
    FileAction,
    FileStatus,
    MetadataStoreError,
    delete_file_metadata,
    evaluate_file_status,
    find_deleted_files,
    mark_file_error,
    upsert_file_metadata,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mongo_collection(monkeypatch):
    client = mongomock.MongoClient()
    col = client["rag"]["files_metadata"]

    monkeypatch.setattr(metadata_store, "_get_collection", lambda: col)
    monkeypatch.setattr(metadata_store, "_index_ensured", True)
    return col


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"Zawartosc pliku testowego")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_compute_file_hash_matches_expected(sample_file):
    expected = _sha256(sample_file.read_bytes())
    assert metadata_store._compute_file_hash(str(sample_file)) == expected


def test_compute_file_hash_raises_for_missing_file(tmp_path):
    with pytest.raises(MetadataStoreError):
        metadata_store._compute_file_hash(str(tmp_path / "nope.pdf"))


def test_evaluate_returns_ADD_when_file_absent_from_store(mongo_collection, sample_file):
    result = evaluate_file_status(str(sample_file))

    assert result.action == FileAction.ADD
    assert result.file_path == str(sample_file)
    assert result.content_hash == _sha256(sample_file.read_bytes())
    assert result.old_parent_doc_ids == []
    assert result.old_child_vector_ids == []


def test_evaluate_returns_SKIP_when_hash_matches(mongo_collection, sample_file):
    content_hash = _sha256(sample_file.read_bytes())
    mongo_collection.insert_one({
        "file_path": str(sample_file),
        "content_hash": content_hash,
        "parent_doc_ids": ["p1"],
        "child_vector_ids": ["c1", "c2"],
        "status": FileStatus.PROCESSED.value,
    })

    result = evaluate_file_status(str(sample_file))
    assert result.action == FileAction.SKIP


def test_evaluate_returns_UPDATE_with_old_ids_when_hash_differs(mongo_collection, sample_file):
    mongo_collection.insert_one({
        "file_path": str(sample_file),
        "content_hash": "stary-hash-sha256",
        "parent_doc_ids": ["p1", "p2"],
        "child_vector_ids": ["c1", "c2", "c3"],
        "status": FileStatus.PROCESSED.value,
    })

    result = evaluate_file_status(str(sample_file))

    assert result.action == FileAction.UPDATE
    assert result.content_hash == _sha256(sample_file.read_bytes())
    assert result.old_parent_doc_ids == ["p1", "p2"]
    assert result.old_child_vector_ids == ["c1", "c2", "c3"]


def test_find_deleted_files_returns_records_not_in_current_list(mongo_collection):
    mongo_collection.insert_many([
        {"file_path": "/a.pdf", "parent_doc_ids": ["p-a"], "child_vector_ids": ["c-a"]},
        {"file_path": "/b.pdf", "parent_doc_ids": ["p-b"], "child_vector_ids": ["c-b1", "c-b2"]},
        {"file_path": "/c.pdf", "parent_doc_ids": ["p-c"], "child_vector_ids": []},
    ])

    deletions = find_deleted_files(["/a.pdf", "/c.pdf"])

    assert len(deletions) == 1
    ev = deletions[0]
    assert ev.action == FileAction.DELETE
    assert ev.file_path == "/b.pdf"
    assert ev.old_parent_doc_ids == ["p-b"]
    assert ev.old_child_vector_ids == ["c-b1", "c-b2"]


def test_find_deleted_files_returns_empty_when_all_present(mongo_collection):
    mongo_collection.insert_one({"file_path": "/a.pdf", "parent_doc_ids": [], "child_vector_ids": []})
    assert find_deleted_files(["/a.pdf"]) == []


def test_find_deleted_files_returns_empty_on_empty_store(mongo_collection):
    assert find_deleted_files(["/a.pdf"]) == []


def test_upsert_inserts_new_record(mongo_collection):
    upsert_file_metadata(
        file_path="/x.pdf",
        content_hash="h1",
        parent_doc_ids=["p1"],
        child_vector_ids=["c1"],
    )

    rec = mongo_collection.find_one({"file_path": "/x.pdf"})
    assert rec is not None
    assert rec["content_hash"] == "h1"
    assert rec["parent_doc_ids"] == ["p1"]
    assert rec["child_vector_ids"] == ["c1"]
    assert rec["status"] == FileStatus.PROCESSED.value
    assert "last_updated" in rec


def test_upsert_updates_existing_record(mongo_collection):
    upsert_file_metadata("/x.pdf", "h1", ["p1"], ["c1"])
    upsert_file_metadata("/x.pdf", "h2", ["p2", "p3"], ["c2"])

    assert mongo_collection.count_documents({"file_path": "/x.pdf"}) == 1
    rec = mongo_collection.find_one({"file_path": "/x.pdf"})
    assert rec["content_hash"] == "h2"
    assert rec["parent_doc_ids"] == ["p2", "p3"]
    assert rec["child_vector_ids"] == ["c2"]


def test_delete_removes_record(mongo_collection):
    upsert_file_metadata("/x.pdf", "h1", [], [])
    delete_file_metadata("/x.pdf")
    assert mongo_collection.count_documents({"file_path": "/x.pdf"}) == 0


def test_mark_file_error_on_existing_record(mongo_collection):
    upsert_file_metadata("/x.pdf", "h1", ["p1"], ["c1"])
    mark_file_error("/x.pdf", content_hash="h2")

    rec = mongo_collection.find_one({"file_path": "/x.pdf"})
    assert rec["status"] == FileStatus.ERROR.value
    assert rec["content_hash"] == "h2"


def test_mark_file_error_creates_record_when_missing(mongo_collection):
    mark_file_error("/x.pdf", content_hash="h1")

    rec = mongo_collection.find_one({"file_path": "/x.pdf"})
    assert rec is not None
    assert rec["status"] == FileStatus.ERROR.value
