import pytest

from src.retrieval.filters import InvalidFilterError, build_qdrant_filter

pytestmark = pytest.mark.unit


def test_none_returns_none():
    assert build_qdrant_filter(None) is None


def test_empty_dict_returns_none():
    assert build_qdrant_filter({}) is None


def test_scalar_value_becomes_match_value():
    from qdrant_client.models import MatchValue

    result = build_qdrant_filter({"filename": "umowa.pdf"})
    assert result is not None
    assert len(result.must) == 1
    cond = result.must[0]
    assert cond.key == "filename"
    assert isinstance(cond.match, MatchValue)
    assert cond.match.value == "umowa.pdf"


def test_list_value_becomes_match_any():
    from qdrant_client.models import MatchAny

    result = build_qdrant_filter({"filename": ["a.pdf", "b.pdf"]})
    cond = result.must[0]
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == ["a.pdf", "b.pdf"]


def test_numeric_range_becomes_range():
    from qdrant_client.models import Range

    result = build_qdrant_filter({"page": {"gte": 5, "lte": 20}})
    cond = result.must[0]
    assert isinstance(cond.range, Range)
    assert cond.range.gte == 5
    assert cond.range.lte == 20


def test_string_range_becomes_datetime_range():
    from qdrant_client.models import DatetimeRange

    result = build_qdrant_filter({
        "ingested_at": {"gte": "2026-01-01T00:00:00+00:00"}
    })
    cond = result.must[0]
    assert isinstance(cond.range, DatetimeRange)


def test_multiple_fields_are_combined_with_must_and():
    result = build_qdrant_filter({
        "filename": "x.pdf",
        "file_extension": "pdf",
    })
    assert len(result.must) == 2


def test_unknown_field_raises():
    with pytest.raises(InvalidFilterError, match="nie jest dozwolone"):
        build_qdrant_filter({"random_field": "x"})


def test_unknown_operator_in_range_raises():
    with pytest.raises(InvalidFilterError, match="Nieznane operatory"):
        build_qdrant_filter({"page": {"equals": 5}})


def test_empty_range_raises():
    with pytest.raises(InvalidFilterError, match="Pusty zakres"):
        build_qdrant_filter({"page": {}})


def test_empty_list_raises():
    with pytest.raises(InvalidFilterError, match="Pusta lista"):
        build_qdrant_filter({"filename": []})


def test_allowed_fields_cover_payload_schema():
    """Sanity check — jeśli dodamy nowe pole do payloadu, musimy je dopisać też tutaj."""
    from src.retrieval.filters import ALLOWED_FILTER_FIELDS

    assert ALLOWED_FILTER_FIELDS == {
        "source", "filename", "file_extension", "page", "ingested_at", "parent_id",
        "article_id", "article_date", "article_title",
    }
