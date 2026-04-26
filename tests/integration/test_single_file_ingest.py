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
    assert files_after == files_before  # nie dotknęło drugiego pliku
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


def test_endpoint_ingest_file_first_time(client, copy_fixture, source_dir, monkeypatch):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    path = copy_fixture("rag_systems_pl.txt")

    response = client.post("/ingest/file/", json={"path": str(path)})

    assert response.status_code == 200
    data = response.json()
    assert data["replaced_existing"] is False
    assert data["parents_count"] > 0
    assert data["children_count"] > 0
    assert data["strategy"]


def test_endpoint_ingest_file_second_time_replaces(client, copy_fixture, source_dir, monkeypatch):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    path = copy_fixture("rag_systems_pl.txt")
    client.post("/ingest/file/", json={"path": str(path)})

    response = client.post("/ingest/file/", json={"path": str(path)})

    assert response.status_code == 200
    assert response.json()["replaced_existing"] is True


def test_endpoint_rejects_path_outside_source_dir(client, source_dir, monkeypatch, tmp_path):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    outside = tmp_path / "outside.pdf"
    outside.write_text("dummy")

    response = client.post("/ingest/file/", json={"path": str(outside)})

    assert response.status_code == 400
    assert "DOKUMENTY" in response.json()["detail"] or "wewnątrz" in response.json()["detail"]


def test_endpoint_returns_404_for_missing_file(client, source_dir, monkeypatch):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    source_dir.mkdir(parents=True, exist_ok=True)
    missing = source_dir / "nieistniejacy.pdf"

    response = client.post("/ingest/file/", json={"path": str(missing)})

    assert response.status_code == 404


def test_endpoint_strategy_override_passed_through(client, copy_fixture, source_dir, monkeypatch):
    from src import config
    from src.server import routes

    monkeypatch.setattr(config, "PDF_SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(routes, "PDF_SOURCE_DIR", str(source_dir))

    path = copy_fixture("rag_systems_pl.txt")

    # .txt nie używa strategii w realnej ekstrakcji, ale walidujemy że pole leci dalej
    response = client.post(
        "/ingest/file/",
        json={"path": str(path), "strategy": "fast"},
    )

    assert response.status_code == 200
    assert response.json()["strategy"] == "fast"
