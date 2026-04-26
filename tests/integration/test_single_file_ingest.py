import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from src.server.app import app
    return TestClient(app)


def _counts():
    from src.config import COLLECTION_NAME, MONGODB_DB, MONGODB_FILES_METADATA_COLLECTION
    from src.db.client import get_client
    from src.db.mongo import get_mongo_client, get_parents_collection

    qdrant = get_client()
    try:
        info = qdrant.get_collection(collection_name=COLLECTION_NAME)
        vectors = info.points_count
    except Exception:
        vectors = 0

    parents = get_parents_collection().count_documents({})
    files = get_mongo_client()[MONGODB_DB][MONGODB_FILES_METADATA_COLLECTION].count_documents({})
    return vectors, parents, files


def test_run_single_file_first_time_returns_added_not_replaced(copy_fixture, source_dir):
    from src.ingest.pipeline import run_single_file

    path = copy_fixture("rag_systems_pl.txt")

    result = run_single_file(str(path))

    assert result.replaced_existing is False
    assert len(result.parent_ids) > 0
    assert len(result.child_ids) > 0


def test_run_single_file_second_time_replaces_old_data(copy_fixture, source_dir):
    from src.ingest.pipeline import run_single_file

    path = copy_fixture("rag_systems_pl.txt")

    first = run_single_file(str(path))
    second = run_single_file(str(path))

    assert second.replaced_existing is True
    assert set(first.parent_ids).isdisjoint(set(second.parent_ids))
    assert set(first.child_ids).isdisjoint(set(second.child_ids))

    _, parents, files = _counts()
    assert files == 1
    assert parents == len(second.parent_ids)


def test_run_single_file_does_not_touch_other_files(copy_fixture, source_dir):
    from src.ingest.pipeline import run_single_file, run_sync

    copy_fixture("machine_learning_pl.txt")
    target = copy_fixture("rag_systems_pl.txt")
    run_sync(source_dir=str(source_dir))

    _, parents_before, files_before = _counts()

    run_single_file(str(target))

    _, parents_after, files_after = _counts()
    assert files_after == files_before
    assert parents_after > 0


def test_run_single_file_missing_path_raises_filenotfound():
    from src.ingest.pipeline import run_single_file

    with pytest.raises(FileNotFoundError):
        run_single_file("/tmp/this_definitely_does_not_exist_zzz.pdf")


def test_run_single_file_unsupported_extension_raises_valueerror(tmp_path):
    from src.ingest.pipeline import run_single_file

    junk = tmp_path / "data.json"
    junk.write_text("{}")

    with pytest.raises(ValueError, match="Nieobsługiwane rozszerzenie"):
        run_single_file(str(junk))


def test_run_sync_paths_processes_only_listed(copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync_paths

    a = copy_fixture("machine_learning_pl.txt")
    b = copy_fixture("rag_systems_pl.txt")

    result = run_sync_paths([str(a)])

    assert result.added == 1
    assert result.updated == 0
    assert result.errors == []

    # Drugi plik nie powinien zostać dotknięty
    _, _, files = _counts()
    assert files == 1


def test_run_sync_paths_dedupes_and_replaces_on_repeat(copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync_paths

    p = copy_fixture("rag_systems_pl.txt")

    # Pierwszy raz: ADD; duplikat ścieżki w wejściu — dedupe → 1 added
    first = run_sync_paths([str(p), str(p)])
    assert first.added == 1
    assert first.updated == 0

    # Drugi raz: UPD bo już jest
    second = run_sync_paths([str(p)])
    assert second.added == 0
    assert second.updated == 1


def test_run_sync_paths_collects_errors_for_missing_files(copy_fixture):
    from src.ingest.pipeline import run_sync_paths

    valid = copy_fixture("rag_systems_pl.txt")
    result = run_sync_paths([str(valid), "/tmp/no_such_file_xyz.pdf"])

    assert result.added == 1
    assert len(result.errors) == 1
    assert "no_such_file_xyz.pdf" in result.errors[0].path


def test_endpoint_sync_with_paths_first_time(client, copy_fixture, source_dir, monkeypatch):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    path = copy_fixture("rag_systems_pl.txt")

    response = client.post("/ingest/sync/", json={"paths": [str(path)]})

    assert response.status_code == 200
    data = response.json()
    assert data["added"] == 1
    assert data["updated"] == 0
    assert data["errors"] == []
    assert data["strategy"]


def test_endpoint_sync_with_paths_second_time_updates(client, copy_fixture, source_dir, monkeypatch):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    path = copy_fixture("rag_systems_pl.txt")
    client.post("/ingest/sync/", json={"paths": [str(path)]})

    response = client.post("/ingest/sync/", json={"paths": [str(path)]})

    assert response.status_code == 200
    assert response.json()["updated"] == 1
    assert response.json()["added"] == 0


def test_endpoint_sync_rejects_paths_outside_source_dir(client, source_dir, monkeypatch, tmp_path):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    outside = tmp_path / "outside.pdf"
    outside.write_text("dummy")

    response = client.post("/ingest/sync/", json={"paths": [str(outside)]})

    assert response.status_code == 400


def test_endpoint_sync_no_paths_runs_full_sync(client, copy_fixture, source_dir, monkeypatch):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    copy_fixture("rag_systems_pl.txt")
    copy_fixture("machine_learning_pl.txt")

    response = client.post("/ingest/sync/", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["added"] == 2
    assert data["errors"] == []


def test_endpoint_sync_strategy_override(client, copy_fixture, source_dir, monkeypatch):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    path = copy_fixture("rag_systems_pl.txt")
    response = client.post(
        "/ingest/sync/",
        json={"paths": [str(path)], "strategy": "fast"},
    )

    assert response.status_code == 200
    assert response.json()["strategy"] == "fast"


def test_endpoint_rebuild_without_confirm_returns_400(client):
    response = client.post("/ingest/rebuild/", json={})

    assert response.status_code in (400, 422)


def test_endpoint_rebuild_with_wrong_confirm_returns_400(client):
    response = client.post("/ingest/rebuild/", json={"confirm": "yes"})

    assert response.status_code == 400


def test_endpoint_rebuild_with_correct_confirm_runs_full_rebuild(client, copy_fixture, source_dir, monkeypatch):
    from src import config
    from src.server import routes
    from src.ingest import pipeline

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(pipeline, "PDF_SOURCE_DIR", str(source_dir))

    copy_fixture("rag_systems_pl.txt")
    copy_fixture("machine_learning_pl.txt")

    response = client.post("/ingest/rebuild/", json={"confirm": "DELETE_ALL"})

    assert response.status_code == 200
    data = response.json()
    assert data["added"] == 2
    assert data["errors"] == []
