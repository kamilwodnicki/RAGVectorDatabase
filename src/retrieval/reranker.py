import logging
from dataclasses import dataclass

from src.config import (
    RERANKER_DEVICE,
    RERANKER_MAX_PARENT_CHARS,
    RERANKER_MODEL,
)

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """Wejście do rerankera: ID parenta, jego tekst, oraz score z retrievalu
    (RRF z fuzji dense+sparse). Score retrievalu zostaje przeniesiony do
    output'u, żeby można było logować obie metryki."""
    parent_id: str
    text: str
    retrieval_score: float = 0.0


@dataclass
class RerankResult:
    parent_id: str
    rerank_score: float
    retrieval_score: float


class Reranker:
    """Cross-encoder reranker. Lazy-load modelu przy pierwszym `rerank()`.

    Przekazuje pary (query, parent_text) do modelu, zwraca listę posortowaną
    malejąco po rerank_score. Tekst parenta jest przycinany do
    RERANKER_MAX_PARENT_CHARS, żeby nie przekroczyć max_length cross-encodera.
    """

    def __init__(self, model_name: str = RERANKER_MODEL, device: str = RERANKER_DEVICE):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Ładowanie rerankera: %s na %s", self.model_name, self.device)
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=512,
        )
        logger.info("Reranker załadowany.")

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
    ) -> list[RerankResult]:
        if not candidates:
            return []
        self._load()
        pairs = [
            [query, c.text[:RERANKER_MAX_PARENT_CHARS]]
            for c in candidates
        ]
        scores = self._model.predict(pairs)
        results = [
            RerankResult(
                parent_id=c.parent_id,
                rerank_score=float(s),
                retrieval_score=c.retrieval_score,
            )
            for c, s in zip(candidates, scores)
        ]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results
