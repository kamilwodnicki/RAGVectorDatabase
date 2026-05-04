import pytest

pytestmark = pytest.mark.integration


def _get_collection_name():
    from src.config import COLLECTION_NAME
    return COLLECTION_NAME


def _counts():
    from src.config import MONGODB_DB, MONGODB_FILES_METADATA_COLLECTION
    from src.db.client import get_client
    from src.db.mongo import get_mongo_client, get_parents_collection

    qdrant = get_client()
    try:
        col_info = qdrant.get_collection(collection_name=_get_collection_name())
        vectors = col_info.points_count
    except Exception:
        vectors = 0

    mongo = get_mongo_client()
    parents = get_parents_collection().count_documents({})
    files = mongo[MONGODB_DB][MONGODB_FILES_METADATA_COLLECTION].count_documents({})
    return vectors, parents, files


def _metadata_record(file_path: str):
    from src.config import MONGODB_DB, MONGODB_FILES_METADATA_COLLECTION
    from src.db.mongo import get_mongo_client

    return get_mongo_client()[MONGODB_DB][MONGODB_FILES_METADATA_COLLECTION].find_one(
        {"file_path": file_path}
    )


def test_add_action_persists_parents_children_and_metadata(copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync

    path = copy_fixture("machine_learning_pl.txt")

    run_sync(source_dir=str(source_dir))

    vectors, parents, files = _counts()
    assert vectors > 0
    assert parents > 0
    assert files == 1

    record = _metadata_record(str(path))
    assert record is not None
    assert record["status"] == "PROCESSED"
    assert len(record["parent_doc_ids"]) == parents
    assert len(record["child_vector_ids"]) == vectors


def test_skip_action_when_file_unchanged(copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync

    copy_fixture("machine_learning_pl.txt")

    run_sync(source_dir=str(source_dir))
    first = _counts()

    run_sync(source_dir=str(source_dir))
    second = _counts()

    assert first == second


def test_update_action_replaces_old_vectors_and_parents(copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync

    path = copy_fixture("machine_learning_pl.txt")

    run_sync(source_dir=str(source_dir))
    old_record = _metadata_record(str(path))
    old_parent_ids = set(old_record["parent_doc_ids"])
    old_child_ids = set(old_record["child_vector_ids"])

    path.write_text(
        "Zupełnie nowa treść dokumentu po modyfikacji.\n\n"
        "Druga sekcja zawiera inne słowa i zdania niż wcześniej, "
        "a model powinien wygenerować zupełnie nowe embeddingi.",
        encoding="utf-8",
    )

    run_sync(source_dir=str(source_dir))

    new_record = _metadata_record(str(path))
    new_parent_ids = set(new_record["parent_doc_ids"])
    new_child_ids = set(new_record["child_vector_ids"])

    assert new_parent_ids.isdisjoint(old_parent_ids)
    assert new_child_ids.isdisjoint(old_child_ids)
    assert new_record["content_hash"] != old_record["content_hash"]

    _, _, files_count = _counts()
    assert files_count == 1


def test_delete_action_removes_file_from_all_stores(copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync

    path = copy_fixture("machine_learning_pl.txt")
    copy_fixture("rag_systems_pl.txt")

    run_sync(source_dir=str(source_dir))
    _, parents_before, files_before = _counts()
    assert files_before == 2

    path.unlink()

    run_sync(source_dir=str(source_dir))

    _, parents_after, files_after = _counts()
    assert files_after == 1
    assert parents_after < parents_before
    assert _metadata_record(str(path)) is None


def test_delete_action_removes_child_vectors_from_qdrant(copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync

    path = copy_fixture("machine_learning_pl.txt")
    copy_fixture("rag_systems_pl.txt")

    run_sync(source_dir=str(source_dir))

    deleted_record = _metadata_record(str(path))
    deleted_vector_ids = deleted_record["child_vector_ids"]
    assert len(deleted_vector_ids) > 0

    vectors_before, _, _ = _counts()

    path.unlink()
    run_sync(source_dir=str(source_dir))

    vectors_after, _, _ = _counts()
    assert vectors_after == vectors_before - len(deleted_vector_ids)

    from src.db.client import get_client
    qdrant = get_client()
    found = qdrant.retrieve(
        collection_name=_get_collection_name(),
        ids=deleted_vector_ids,
    )
    assert found == []
