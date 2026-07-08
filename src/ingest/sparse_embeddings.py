import re
from dataclasses import dataclass
from functools import lru_cache

from src.config import SPARSE_MODEL_NAME, SPARSE_STEMMER


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class _PolishStemmer:
    def __init__(self):
        # pystempel 2.x: import `pystempel.Stemmer`, obiekt jest wołalny (stemmer(word)).
        from pystempel import Stemmer

        self._stemmer = Stemmer.default()
        self._stem_cached = lru_cache(maxsize=200_000)(self._stem_one)

    def _stem_one(self, token: str) -> str:
        stem = self._stemmer(token)
        return stem if stem else token

    def stem_text(self, text: str) -> str:
        return " ".join(self._stem_cached(t) for t in _TOKEN_RE.findall(text.lower()))


class BM25SparseEmbeddings:
    """
    Wrapper na fastembed.SparseTextEmbedding (BM25).

    Rozróżnia .embed_documents() (pełny BM25 z IDF) od .embed_query()
    (query-side weighting bez IDF) — tak działa konwencjonalny BM25.
    """

    def __init__(self, model_name: str | None = None):
        from fastembed import SparseTextEmbedding

        self._stemmer = _PolishStemmer() if SPARSE_STEMMER == "stempel" else None
        model_kwargs = {"model_name": model_name or SPARSE_MODEL_NAME}
        if self._stemmer is not None:
            # nie stemuj dwa razy (i nie po angielsku)
            model_kwargs["disable_stemmer"] = True
        self._model = SparseTextEmbedding(**model_kwargs)

    def _prep(self, text: str) -> str:
        return self._stemmer.stem_text(text) if self._stemmer is not None else text

    def embed_documents(self, texts: list[str]) -> list[SparseVectorData]:
        prepared = [self._prep(t) for t in texts]
        return [self._to_data(e) for e in self._model.embed(prepared)]

    def embed_query(self, text: str) -> SparseVectorData:
        prepared = self._prep(text)
        return self._to_data(next(iter(self._model.query_embed([prepared]))))

    @staticmethod
    def _to_data(emb) -> SparseVectorData:
        return SparseVectorData(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
