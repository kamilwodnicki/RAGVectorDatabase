from pydantic import BaseModel

from src.config import DEFAULT_K


class QueryRequest(BaseModel):
    query: str
    k: int = DEFAULT_K
    filters: dict | None = None


class DocumentFragment(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    results: list[DocumentFragment]


class IngestFileRequest(BaseModel):
    path: str
    strategy: str | None = None  # None → użyj EXTRACTION_STRATEGY z env


class IngestFileResponse(BaseModel):
    path: str
    strategy: str
    parents_count: int
    children_count: int
    replaced_existing: bool
