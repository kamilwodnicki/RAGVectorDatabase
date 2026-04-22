from dataclasses import dataclass

from src.config import SPARSE_MODEL_NAME


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]


class BM25SparseEmbeddings:
    """
    Wrapper na fastembed.SparseTextEmbedding (BM25).

    Rozróżnia .embed_documents() (pełny BM25 z IDF) od .embed_query()
    (query-side weighting bez IDF) — tak działa konwencjonalny BM25.
    """

    def __init__(self, model_name: str | None = None):
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model_name or SPARSE_MODEL_NAME)

    def embed_documents(self, texts: list[str]) -> list[SparseVectorData]:
        return [self._to_data(e) for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVectorData:
        return self._to_data(next(iter(self._model.query_embed([text]))))

    @staticmethod
    def _to_data(emb) -> SparseVectorData:
        return SparseVectorData(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
