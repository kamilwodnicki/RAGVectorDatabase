from pathlib import Path
from langchain_core.documents import Document


def extract_text(path: Path) -> list[Document]:
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return []
    return [Document(page_content=content, metadata={"source": str(path)})]
