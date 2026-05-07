import uuid
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.documents import Document
from src.config import (
    PARENT_MAX_SIZE,
    PARENT_SOFT_SIZE,
    PARENT_COMBINE_UNDER,
    PARENT_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)


def chunk_file_elements(
    path: Path,
    elements,
    extra_metadata: dict | None = None,
) -> tuple[list[dict], list[Document]]:
    from unstructured.chunking.title import chunk_by_title
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_chunks = chunk_by_title(
        elements,
        max_characters=PARENT_MAX_SIZE,
        new_after_n_chars=PARENT_SOFT_SIZE,
        combine_text_under_n_chars=PARENT_COMBINE_UNDER,
        overlap=PARENT_OVERLAP,
        overlap_all=False,
        multipage_sections=True,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    source = str(path)
    filename = path.name
    file_extension = path.suffix.lower().lstrip(".")
    ingested_at = datetime.now(timezone.utc).isoformat()
    extra = extra_metadata or {}

    parents: list[dict] = []
    children: list[Document] = []

    for pc in parent_chunks:
        parent_id = str(uuid.uuid4())
        page = getattr(pc.metadata, "page_number", None)

        parent_doc = {
            "_id": parent_id,
            "text": pc.text,
            "source": source,
            "filename": filename,
            "file_extension": file_extension,
            "page": page,
            "ingested_at": ingested_at,
            **extra,
        }
        parents.append(parent_doc)

        child_meta_base = {
            "parent_id": parent_id,
            "source": source,
            "filename": filename,
            "file_extension": file_extension,
            "page": page,
            "ingested_at": ingested_at,
            **extra,
        }
        for child_text in child_splitter.split_text(pc.text):
            children.append(Document(
                page_content=child_text,
                metadata=dict(child_meta_base),
            ))

    return parents, children
