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


class IngestSyncRequest(BaseModel):
    paths: list[str] | None = None  # None/[] → sync całego korpusu (incremental)
    strategy: str | None = None     # None → użyj EXTRACTION_STRATEGY z env


class IngestRebuildRequest(BaseModel):
    confirm: str                     # MUSI równać się "DELETE_ALL", inaczej HTTP 400
    strategy: str | None = None


class SyncErrorItem(BaseModel):
    path: str
    error: str


class IngestSyncResponse(BaseModel):
    added: int
    updated: int
    skipped: int
    deleted: int
    errors: list[SyncErrorItem]
    elapsed_seconds: float
    strategy: str
