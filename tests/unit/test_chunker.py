from pathlib import Path

import pytest

from src.config import CHILD_CHUNK_SIZE, PARENT_MAX_SIZE
from src.ingest.chunker import chunk_file_elements

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_path():
    return Path("DOKUMENTY/fake.pdf")


def _make_elements(*texts):
    from unstructured.documents.elements import NarrativeText, Title

    elements = []
    for i, t in enumerate(texts):
        if i == 0 or t.startswith("# "):
            elements.append(Title(t.lstrip("# ")))
        else:
            elements.append(NarrativeText(t))
    return elements


def test_empty_elements_returns_empty_lists(fake_path):
    parents, children = chunk_file_elements(fake_path, [])
    assert parents == []
    assert children == []


def test_short_section_produces_single_parent_with_one_or_more_children(fake_path):
    elements = _make_elements(
        "Sekcja pierwsza",
        "Krótki akapit treści merytorycznej, kilka zdań."
    )
    parents, children = chunk_file_elements(fake_path, elements)

    assert len(parents) == 1
    assert len(children) >= 1


def test_parent_has_required_fields(fake_path):
    elements = _make_elements("Tytuł", "Treść akapitu.")
    parents, _ = chunk_file_elements(fake_path, elements)

    p = parents[0]
    assert set(p.keys()) >= {
        "_id", "text", "source", "filename", "file_extension", "page", "ingested_at",
    }
    assert isinstance(p["_id"], str) and len(p["_id"]) > 0
    assert p["source"] == str(fake_path)
    assert p["filename"] == "fake.pdf"
    assert p["file_extension"] == "pdf"
    assert p["text"]
    assert p["ingested_at"]


def test_every_child_carries_parent_id(fake_path):
    elements = _make_elements("Tytuł", "Treść akapitu pierwszego.", "Treść akapitu drugiego.")
    parents, children = chunk_file_elements(fake_path, elements)

    parent_ids = {p["_id"] for p in parents}
    assert len(children) > 0
    for child in children:
        assert "parent_id" in child.metadata
        assert child.metadata["parent_id"] in parent_ids


def test_every_child_carries_source(fake_path):
    elements = _make_elements("Tytuł", "Treść.")
    _, children = chunk_file_elements(fake_path, elements)

    for child in children:
        assert child.metadata["source"] == str(fake_path)


def test_every_child_carries_file_level_metadata(fake_path):
    elements = _make_elements("Tytuł", "Treść.")
    _, children = chunk_file_elements(fake_path, elements)

    assert len(children) > 0
    for child in children:
        assert child.metadata["filename"] == "fake.pdf"
        assert child.metadata["file_extension"] == "pdf"
        assert child.metadata["ingested_at"]


def test_txt_file_extension_is_normalized_lowercase_without_dot(tmp_path):
    elements = _make_elements("Tytuł", "Treść.")
    parents, children = chunk_file_elements(tmp_path / "Notes.TXT", elements)

    assert parents[0]["file_extension"] == "txt"
    for child in children:
        assert child.metadata["file_extension"] == "txt"


def test_all_chunks_of_single_file_share_ingested_at(fake_path):
    long_text = " ".join(["słowo"] * 400)
    elements = _make_elements("Sekcja A", long_text, "# Sekcja B", long_text)
    parents, children = chunk_file_elements(fake_path, elements)

    timestamps = {p["ingested_at"] for p in parents} | {
        c.metadata["ingested_at"] for c in children
    }
    assert len(timestamps) == 1


def test_parent_ids_are_unique(fake_path):
    long_text = " ".join(["słowo"] * 400)
    elements = _make_elements(
        "Sekcja A", long_text,
        "# Sekcja B", long_text,
        "# Sekcja C", long_text,
    )
    parents, _ = chunk_file_elements(fake_path, elements)

    ids = [p["_id"] for p in parents]
    assert len(ids) == len(set(ids))


def test_parent_respects_max_size(fake_path):
    long_text = " ".join(["słowo"] * 800)
    elements = _make_elements("Sekcja", long_text)
    parents, _ = chunk_file_elements(fake_path, elements)

    for p in parents:
        assert len(p["text"]) <= PARENT_MAX_SIZE


def test_child_size_stays_close_to_configured_limit(fake_path):
    long_text = " ".join(["słowo"] * 800)
    elements = _make_elements("Sekcja", long_text)
    _, children = chunk_file_elements(fake_path, elements)

    tolerance = 1.5
    for child in children:
        assert len(child.page_content) <= int(CHILD_CHUNK_SIZE * tolerance)


def test_new_title_preferably_starts_new_parent_when_content_is_large_enough(fake_path):
    section_a_body = " ".join(["alpha"] * 200)
    section_b_body = " ".join(["beta"] * 200)
    elements = _make_elements(
        "Sekcja A", section_a_body,
        "# Sekcja B", section_b_body,
    )
    parents, _ = chunk_file_elements(fake_path, elements)

    assert len(parents) >= 2
    joined = " ".join(p["text"] for p in parents)
    assert "alpha" in joined and "beta" in joined
