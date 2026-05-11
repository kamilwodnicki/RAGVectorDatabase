import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_json_article(path: Path) -> tuple[list, dict] | tuple[None, None]:
    """Czyta artykuł w formacie JSON {id, date, title, content}.

    Zwraca:
        (elements, extra_meta)  — lista elementów unstructured + metadata artykułu
        (None, None)            — content pusty → ingest powinien skipnąć

    Tytuł (jeśli niepusty) zostaje doklejony jako prefiks treści w jednym
    NarrativeText. Osobny element Title powodował, że chunk_by_title tworzył
    samotny parent zawierający tylko tytuł — szum w retrievalu.
    """
    from unstructured.documents.elements import NarrativeText

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    content = (data.get("content") or "").strip()
    if not content:
        return None, None

    title = (data.get("title") or "").strip()
    text = f"{title}\n\n{content}" if title else content
    elements: list = [NarrativeText(text=text)]

    extra_meta = {
        "article_id": str(data["id"]) if data.get("id") is not None else None,
        "article_date": data.get("date"),
        "article_title": title or None,
    }
    return elements, extra_meta
