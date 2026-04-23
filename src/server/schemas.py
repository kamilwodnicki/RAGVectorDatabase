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
