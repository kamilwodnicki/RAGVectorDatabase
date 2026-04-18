from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    k: int = 3


class DocumentFragment(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    results: list[DocumentFragment]
