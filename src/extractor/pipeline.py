from pathlib import Path
from src.extractor.pdf import extract_pdf
from src.extractor.text import extract_text
from src.extractor.cleaner import clean
from src.config import EXTRACTION_STRATEGY


def extract_single_file(path: Path, strategy: str = EXTRACTION_STRATEGY):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        elements = extract_pdf(path, strategy=strategy)
    elif suffix == ".txt":
        elements = extract_text(path)
    else:
        return []

    cleaned = []
    for el in elements:
        el.text = clean(el.text)
        if el.text:
            cleaned.append(el)
    return cleaned


def extract_documents_per_file(source_dir: str, strategy: str = EXTRACTION_STRATEGY):
    results = []
    for path in sorted(Path(source_dir).rglob("*")):
        if not path.is_file():
            continue
        cleaned = extract_single_file(path, strategy=strategy)
        if cleaned:
            results.append((path, cleaned))
    return results
