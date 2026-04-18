from pathlib import Path
from langchain_core.documents import Document
from src.extractor.pdf import extract_pdf
from src.extractor.text import extract_text
from src.extractor.cleaner import clean
from src.config import EXTRACTION_STRATEGY


def extract_documents(source_dir: str, strategy: str = EXTRACTION_STRATEGY) -> list[Document]:
    documents = []
    for path in sorted(Path(source_dir).rglob("*")):
        if path.suffix.lower() == ".pdf":
            raw = extract_pdf(path, strategy=strategy)
        elif path.suffix.lower() == ".txt":
            raw = extract_text(path)
        else:
            continue
        for doc in raw:
            doc.page_content = clean(doc.page_content)
            if doc.page_content:
                documents.append(doc)
    return documents
