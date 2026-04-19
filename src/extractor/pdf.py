import logging
from pathlib import Path
from src.config import EXTRACTION_LANGUAGES


class _NoFeaturesFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "No features in text" not in record.getMessage()


logging.getLogger("unstructured").addFilter(_NoFeaturesFilter())


def extract_pdf(path: Path, strategy: str = "fast"):
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import NarrativeText, Title, ListItem, Table

    keep = (NarrativeText, Title, ListItem, Table)
    elements = partition_pdf(
        str(path),
        strategy=strategy,
        languages=EXTRACTION_LANGUAGES,
    )
    return [el for el in elements if isinstance(el, keep)]
