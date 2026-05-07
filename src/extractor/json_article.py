import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_json_article(path: Path) -> tuple[list, dict] | tuple[None, None]:
    """Czyta artykuł w formacie JSON {id, date, title, content}.

    Zwraca:
        (elements, extra_meta)  — lista elementów unstructured + metadata artykułu
        (None, None)            — content pusty → ingest powinien skipnąć

    Pole `content` jest traktowane jako czysty tekst i opakowane w NarrativeText.
    `title` (jeśli niepusty) leci jako osobny element Title — daje chunk_by_title
    naturalny punkt podziału. Jeśli JSON jest niepoprawny / brakuje pól → wyjątek.
    """
    from unstructured.documents.elements import NarrativeText, Title

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    content = (data.get("content") or "").strip()
    if not content:
        return None, None

    elements: list = []
    title = (data.get("title") or "").strip()
    if title:
        elements.append(Title(text=title))
    elements.append(NarrativeText(text=content))

    extra_meta = {
        "article_id": str(data["id"]) if data.get("id") is not None else None,
        "article_date": data.get("date"),
        "article_title": title or None,
    }
    return elements, extra_meta
