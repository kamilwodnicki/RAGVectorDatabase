import pytest

pytestmark = pytest.mark.unit


def test_weighted_rrf_prefers_document_appearing_in_both_rankings():
    from src.retrieval.hybrid import weighted_rrf

    dense_ranking = ["doc_a", "doc_b", "doc_c"]
    sparse_ranking = ["doc_x", "doc_a", "doc_y"]

    result = weighted_rrf(dense_ranking, sparse_ranking, k=3, rrf_k=60)
    ids = [cid for cid, _ in result]

    assert ids[0] == "doc_a"


def test_weighted_rrf_equal_weights_reproduces_standard_rrf_math():
    from src.retrieval.hybrid import weighted_rrf

    dense_ranking = ["doc_a", "doc_b"]
    sparse_ranking = ["doc_b", "doc_a"]

    result = dict(weighted_rrf(dense_ranking, sparse_ranking, k=10,
                               dense_weight=1.0, sparse_weight=1.0, rrf_k=60))

    # Oba dokumenty mają pozycje (1, 2) i (2, 1) — symetrycznie. Te same score.
    assert result["doc_a"] == pytest.approx(result["doc_b"])
    expected = 1 / 61 + 1 / 62
    assert result["doc_a"] == pytest.approx(expected)


def test_weighted_rrf_sparse_weight_boosts_sparse_only_hits():
    from src.retrieval.hybrid import weighted_rrf

    dense_ranking = ["doc_dense"]
    sparse_ranking = ["doc_sparse"]

    result = dict(weighted_rrf(
        dense_ranking, sparse_ranking,
        k=10, dense_weight=1.0, sparse_weight=5.0, rrf_k=60,
    ))

    assert result["doc_sparse"] > result["doc_dense"]


def test_weighted_rrf_returns_at_most_k_items():
    from src.retrieval.hybrid import weighted_rrf

    dense_ranking = [f"d{i}" for i in range(20)]
    sparse_ranking = [f"d{i}" for i in range(10, 30)]

    result = weighted_rrf(dense_ranking, sparse_ranking, k=5)

    assert len(result) == 5


def test_weighted_rrf_respects_descending_score_order():
    from src.retrieval.hybrid import weighted_rrf

    dense_ranking = ["a", "b", "c"]
    sparse_ranking = ["a", "b", "c"]

    result = weighted_rrf(dense_ranking, sparse_ranking, k=3)
    scores = [s for _, s in result]

    assert scores == sorted(scores, reverse=True)


def test_weighted_rrf_only_in_dense_still_included():
    from src.retrieval.hybrid import weighted_rrf

    dense_ranking = ["only_dense"]
    sparse_ranking = ["only_sparse"]

    result = dict(weighted_rrf(dense_ranking, sparse_ranking, k=10))

    assert "only_dense" in result
    assert "only_sparse" in result
    # Oba są na pozycji 1 w swoich rankingach, równe wagi → równe score.
    assert result["only_dense"] == pytest.approx(result["only_sparse"])


def test_retrieve_children_rejects_unknown_mode():
    from src.retrieval.hybrid import retrieve_children

    with pytest.raises(ValueError, match="Nieznany RETRIEVAL_MODE"):
        retrieve_children(
            query="q", k=3, mode="nonsense",
            dense_embedder=None, sparse_embedder=None,
        )


def test_retrieve_children_sparse_mode_requires_sparse_embedder():
    from src.retrieval.hybrid import retrieve_children

    with pytest.raises(ValueError, match="sparse_embedder"):
        retrieve_children(
            query="q", k=3, mode="sparse",
            dense_embedder=object(), sparse_embedder=None,
        )


def test_retrieve_children_hybrid_mode_requires_sparse_embedder():
    from src.retrieval.hybrid import retrieve_children

    with pytest.raises(ValueError, match="sparse_embedder"):
        retrieve_children(
            query="q", k=3, mode="hybrid",
            dense_embedder=object(), sparse_embedder=None,
        )
