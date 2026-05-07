from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)

ALLOWED_FILTER_FIELDS = {
    "source",
    "filename",
    "file_extension",
    "page",
    "ingested_at",
    "parent_id",
    "article_id",
    "article_date",
    "article_title",
}

_RANGE_OPS = {"gte", "lte", "gt", "lt"}


class InvalidFilterError(ValueError):
    pass


def build_qdrant_filter(filters: dict | None) -> Filter | None:
    """Mapuje słownik filtrów z API na Qdrant Filter (AND wszystkich warunków).

    Przykłady:
        {"filename": "x.pdf"}                      → MatchValue
        {"filename": ["a.pdf", "b.pdf"]}           → MatchAny (OR)
        {"page": {"gte": 5, "lte": 20}}            → Range (numeric)
        {"ingested_at": {"gte": "2026-01-01..."}}  → DatetimeRange (string → datetime)
    """
    if not filters:
        return None

    conditions = [_build_condition(field, value) for field, value in filters.items()]
    return Filter(must=conditions)


def _build_condition(field: str, value) -> FieldCondition:
    if field not in ALLOWED_FILTER_FIELDS:
        raise InvalidFilterError(
            f"Pole '{field}' nie jest dozwolone w filtrach. "
            f"Dozwolone: {sorted(ALLOWED_FILTER_FIELDS)}"
        )

    if isinstance(value, dict):
        return _build_range_condition(field, value)
    if isinstance(value, list):
        if not value:
            raise InvalidFilterError(f"Pusta lista wartości dla pola '{field}'")
        return FieldCondition(key=field, match=MatchAny(any=value))
    return FieldCondition(key=field, match=MatchValue(value=value))


def _build_range_condition(field: str, ops: dict) -> FieldCondition:
    if not ops:
        raise InvalidFilterError(f"Pusty zakres dla pola '{field}'")

    unknown = set(ops.keys()) - _RANGE_OPS
    if unknown:
        raise InvalidFilterError(
            f"Nieznane operatory dla pola '{field}': {sorted(unknown)}. "
            f"Dozwolone: {sorted(_RANGE_OPS)}"
        )

    is_datetime = any(isinstance(v, str) for v in ops.values())
    if is_datetime:
        return FieldCondition(key=field, range=DatetimeRange(**ops))
    return FieldCondition(key=field, range=Range(**ops))
