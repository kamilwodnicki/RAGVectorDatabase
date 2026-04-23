import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from src.server.app import app
    return TestClient(app)


def test_query_returns_parent_matching_ingested_content(client, copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync

    copy_fixture("rag_systems_pl.txt")
    run_sync(source_dir=str(source_dir))

    response = client.post(
        "/query/",
        json={"query": "Czym jest architektura parent-child w systemach RAG?", "k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0

    first = data["results"][0]
    assert "content" in first
    assert "metadata" in first
    meta = first["metadata"]
    assert meta["parent_id"]
    assert meta["filename"] == "rag_systems_pl.txt"
    assert meta["file_extension"] == "txt"
    assert meta["ingested_at"]

    combined = " ".join(r["content"] for r in data["results"]).lower()
    assert "parent" in combined or "dziec" in combined or "chunking" in combined


def test_query_returns_empty_when_store_empty(client):
    from src.db.collection import setup_collection

    setup_collection(recreate=True)

    response = client.post("/query/", json={"query": "dowolne pytanie", "k": 3})

    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_query_resolves_children_back_to_distinct_parents(client, copy_fixture, source_dir):
    from src.ingest.pipeline import run_sync

    copy_fixture("machine_learning_pl.txt")
    copy_fixture("rag_systems_pl.txt")
    run_sync(source_dir=str(source_dir))

    response = client.post(
        "/query/",
        json={"query": "embeddingi wektorowe reprezentacje", "k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    parent_ids = [r["metadata"]["parent_id"] for r in data["results"]]
    assert len(parent_ids) == len(set(parent_ids))
