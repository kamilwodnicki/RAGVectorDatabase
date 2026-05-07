from pathlib import Path

from src.config import EXTRACTION_STRATEGY
from src.extractor.cleaner import clean
from src.extractor.json_article import extract_json_article
from src.extractor.pdf import extract_pdf
from src.extractor.text import extract_text


def extract_single_file(
    path: Path, strategy: str = EXTRACTION_STRATEGY
) -> tuple[list, dict]:
    """Zwraca (cleaned_elements, extra_meta).

    extra_meta — dict z polami artykułu (article_id/date/title) tylko dla .json.
    Dla .pdf/.txt zwraca pusty dict.

    Dla .json z pustym contentem zwraca ([], {}) — wywołujący traktuje to jako skip.
    Dla nieobsługiwanego rozszerzenia zwraca ([], {}).
    """
    suffix = path.suffix.lower()
    extra_meta: dict = {}

    if suffix == ".pdf":
        elements = extract_pdf(path, strategy=strategy)
    elif suffix == ".txt":
        elements = extract_text(path)
    elif suffix == ".json":
        elements, extra_meta = extract_json_article(path)
        if elements is None:
            return [], {}
    else:
        return [], {}

    cleaned = []
    for el in elements:
        el.text = clean(el.text)
        if el.text:
            cleaned.append(el)
    return cleaned, extra_meta
