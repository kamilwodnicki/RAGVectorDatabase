from pathlib import Path

import pytest

pytestmark = pytest.mark.eval

HIT_AT_3_THRESHOLD = 0.70
MRR_AT_10_THRESHOLD = 0.55
CHILDREN_FETCHED = 50


def _retrieve_parents(query: str, k_children: int, embeddings) -> list[dict]:
    from langchain_qdrant import QdrantVectorStore

    from src.config import COLLECTION_NAME
    from src.db.client import get_client
    from src.db.mongo import get_parents_collection

    vs = QdrantVectorStore(
        client=get_client(),
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    children = vs.similarity_search(query, k=k_children)

    seen: set[str] = set()
    ordered_parent_ids: list[str] = []
    for child in children:
        pid = child.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            ordered_parent_ids.append(pid)

    if not ordered_parent_ids:
        return []

    parent_map = {
        p["_id"]: p
        for p in get_parents_collection().find({"_id": {"$in": ordered_parent_ids}})
    }
    return [parent_map[pid] for pid in ordered_parent_ids if pid in parent_map]


def _matches(parent: dict, expected: dict) -> bool:
    if Path(parent["source"]).name != expected["expected_source"]:
        return False
    text_lower = parent["text"].lower()
    return any(kw.lower() in text_lower for kw in expected["expected_keywords"])


def _rank_of_match(parents: list[dict], expected: dict) -> int | None:
    for i, p in enumerate(parents, start=1):
        if _matches(p, expected):
            return i
    return None


def test_retrieval_quality_meets_thresholds(ingested_corpus, golden_set, shared_embeddings, capsys):
    results = []
    for q in golden_set:
        parents = _retrieve_parents(
            q["question"], k_children=CHILDREN_FETCHED, embeddings=shared_embeddings
        )
        rank = _rank_of_match(parents, q)
        results.append((q["id"], q["question"], rank))

    n = len(results)
    hit_at_1 = sum(1 for _, _, r in results if r == 1) / n
    hit_at_3 = sum(1 for _, _, r in results if r is not None and r <= 3) / n
    hit_at_5 = sum(1 for _, _, r in results if r is not None and r <= 5) / n
    mrr_at_10 = sum(1 / r for _, _, r in results if r is not None and r <= 10) / n

    report_lines = ["", "=" * 78]
    report_lines.append(f"{'ID':<10} {'rank':<6} pytanie")
    report_lines.append("-" * 78)
    for qid, question, rank in results:
        rank_str = str(rank) if rank is not None else "miss"
        q_short = question if len(question) <= 58 else question[:55] + "..."
        report_lines.append(f"{qid:<10} {rank_str:<6} {q_short}")
    report_lines.append("-" * 78)
    report_lines.append(f"Zapytania:     {n}")
    report_lines.append(f"Hit@1:         {hit_at_1:.2%}")
    report_lines.append(f"Hit@3:         {hit_at_3:.2%}")
    report_lines.append(f"Hit@5:         {hit_at_5:.2%}")
    report_lines.append(f"MRR@10:        {mrr_at_10:.3f}")
    report_lines.append("=" * 78)

    with capsys.disabled():
        print("\n".join(report_lines))

    assert hit_at_3 >= HIT_AT_3_THRESHOLD, (
        f"Hit@3 = {hit_at_3:.2%}, próg >= {HIT_AT_3_THRESHOLD:.0%}"
    )
    assert mrr_at_10 >= MRR_AT_10_THRESHOLD, (
        f"MRR@10 = {mrr_at_10:.3f}, próg >= {MRR_AT_10_THRESHOLD}"
    )
