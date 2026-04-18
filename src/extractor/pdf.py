from pathlib import Path
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Title, ListItem, Table

_KEEP = (NarrativeText, Title, ListItem, Table)


def extract_pdf(path: Path, strategy: str = "fast") -> list[Document]:
    elements = partition_pdf(str(path), strategy=strategy)
    pages: dict[int, list[str]] = {}
    for el in elements:
        if not isinstance(el, _KEEP):
            continue
        page = el.metadata.page_number or 1
        pages.setdefault(page, []).append(str(el))

    return [
        Document(
            page_content="\n\n".join(texts),
            metadata={"source": str(path), "page": page},
        )
        for page, texts in sorted(pages.items())
        if texts
    ]
