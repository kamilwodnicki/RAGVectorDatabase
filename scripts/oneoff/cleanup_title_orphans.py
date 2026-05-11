"""ONE-OFF cleanup: orphan-parenty z bugu w extractor/json_article.py.

Bug: tytuł artykułu leciał jako osobny element Title obok NarrativeText z
contentem. Dla artykułów dłuższych niż ~2000 znaków `chunk_by_title` nie mógł
skleić Title z pierwszym kawałkiem contentu (bo razem przekraczałyby
max_characters), więc Title zostawał samotnym parentem zawierającym tylko
tytuł. Każdy taki orphan ma odpowiadającego child-a w Qdrant z tym samym
~40-znakowym tekstem — śmieć w retrievalu.

Fix w `src/extractor/json_article.py` (tytuł doklejany jako prefiks contentu).
Ten skrypt sprząta dane już w bazie.

Użycie (wewnątrz kontenera rag-server):
    python scripts/oneoff/cleanup_title_orphans.py            # DRY-RUN
    python scripts/oneoff/cleanup_title_orphans.py --apply    # realne usunięcie

Po wykonaniu i weryfikacji — usunąć ten plik z repo.
"""

import argparse
import logging
from collections import defaultdict

from qdrant_client.models import FieldCondition, Filter, MatchAny, PointIdsList

from src.config import (
    COLLECTION_NAME,
    MONGODB_DB,
    MONGODB_FILES_METADATA_COLLECTION,
)
from src.db.client import get_client
from src.db.mongo import get_mongo_client, get_parents_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cleanup-title-orphans")

PARENT_ID_BATCH = 200      # ile parent_ids w jednym scroll-filter MatchAny
SCROLL_LIMIT = 1000
DELETE_BATCH = 1000


def find_orphans(parents_col) -> list[dict]:
    cursor = parents_col.find(
        {
            "file_extension": "json",
            "$expr": {"$eq": ["$text", "$article_title"]},
        },
        {"_id": 1, "source": 1},
    )
    return list(cursor)


def scroll_children_for_parents(client, parent_ids: list[str]) -> list[tuple[str, str]]:
    """Zwraca [(child_id, parent_id), ...] dla wszystkich child-ów,
    których parent_id ∈ parent_ids."""
    results: list[tuple[str, str]] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="parent_id", match=MatchAny(any=parent_ids))],
            ),
            limit=SCROLL_LIMIT,
            offset=next_offset,
            with_payload=["parent_id"],
            with_vectors=False,
        )
        for p in points:
            results.append((str(p.id), p.payload["parent_id"]))
        if next_offset is None:
            break
    return results


def main(apply: bool) -> int:
    parents_col = get_parents_collection()
    files_meta_col = get_mongo_client()[MONGODB_DB][MONGODB_FILES_METADATA_COLLECTION]
    qdrant = get_client()

    log.info("Kolekcja Qdrant: %s | DB Mongo: %s", COLLECTION_NAME, MONGODB_DB)
    log.info("Szukam orphan-parentów (file_extension=json, text == article_title)...")
    orphans = find_orphans(parents_col)
    log.info("Znaleziono %d orphan-parentów", len(orphans))

    if not orphans:
        log.info("Nic do roboty.")
        return 0

    parent_to_source: dict[str, str] = {o["_id"]: o["source"] for o in orphans}
    by_source_parents: dict[str, list[str]] = defaultdict(list)
    for pid, src in parent_to_source.items():
        by_source_parents[src].append(pid)

    all_parent_ids = list(parent_to_source.keys())
    log.info("Dotkniętych plików (unikalnych source): %d", len(by_source_parents))

    log.info("Pobieram child IDs z Qdrant (batch po %d parent_ids)...", PARENT_ID_BATCH)
    child_pairs: list[tuple[str, str]] = []
    for i in range(0, len(all_parent_ids), PARENT_ID_BATCH):
        batch = all_parent_ids[i:i + PARENT_ID_BATCH]
        pairs = scroll_children_for_parents(qdrant, batch)
        child_pairs.extend(pairs)
        done = min(i + PARENT_ID_BATCH, len(all_parent_ids))
        if done % (PARENT_ID_BATCH * 10) == 0 or done == len(all_parent_ids):
            log.info("  ...%d/%d parent_ids; zebranych child IDs: %d",
                     done, len(all_parent_ids), len(child_pairs))

    by_source_children: dict[str, list[str]] = defaultdict(list)
    for child_id, pid in child_pairs:
        src = parent_to_source.get(pid)
        if src:
            by_source_children[src].append(child_id)

    all_child_ids = [c for c, _ in child_pairs]
    log.info("Do usunięcia: %d parents (Mongo) + %d children (Qdrant)",
             len(all_parent_ids), len(all_child_ids))

    if not apply:
        log.info("DRY-RUN — nic nie usuwam.")
        log.info("Przykładowe 3 orphany:")
        for o in orphans[:3]:
            log.info("  _id=%s | source=%s", o["_id"], o.get("source"))
        log.info("Odpal ponownie z --apply żeby wykonać kasowanie.")
        return 0

    log.warning("APPLY — kasuję dane. To jest nieodwracalne (poza rebuildem).")

    log.info("1/3: Aktualizuję files_metadata ($pullAll dla parent_ids i child_ids)...")
    for src, p_ids in by_source_parents.items():
        update = {"$pullAll": {"parent_doc_ids": p_ids}}
        c_ids = by_source_children.get(src)
        if c_ids:
            update["$pullAll"]["child_vector_ids"] = c_ids
        files_meta_col.update_one({"file_path": src}, update)
    log.info("  files_metadata: zaktualizowano %d dokumentów", len(by_source_parents))

    log.info("2/3: Kasuję orphan-parents z Mongo (batchami po %d)...", DELETE_BATCH)
    deleted_parents = 0
    for i in range(0, len(all_parent_ids), DELETE_BATCH):
        batch = all_parent_ids[i:i + DELETE_BATCH]
        result = parents_col.delete_many({"_id": {"$in": batch}})
        deleted_parents += result.deleted_count
    log.info("  Mongo: skasowano %d parents", deleted_parents)

    log.info("3/3: Kasuję child-y z Qdrant (batchami po %d)...", DELETE_BATCH)
    for i in range(0, len(all_child_ids), DELETE_BATCH):
        batch = all_child_ids[i:i + DELETE_BATCH]
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=batch),
        )
    log.info("  Qdrant: skasowano %d points", len(all_child_ids))

    log.info("Gotowe.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true",
        help="Wykonaj realne kasowanie (domyślnie dry-run).",
    )
    args = parser.parse_args()
    raise SystemExit(main(apply=args.apply))
