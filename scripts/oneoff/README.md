# scripts/oneoff/

Jednorazowe skrypty naprawcze. **Nie część normalnego workflow.**

Każdy plik w tym katalogu powinien:
- Mieć w nagłówku komentarz wyjaśniający dla jakiego buga/migracji powstał i kiedy.
- Działać w trybie dry-run domyślnie; realne zmiany pod jawnym `--apply`.
- Być usunięty po wykonaniu (commit historii starczy jako ślad).

## Aktualne skrypty

- `cleanup_title_orphans.py` — czyści orphan-parenty pozostawione przez buga
  w `extractor/json_article.py`, gdzie tytuł artykułu leciał jako osobny
  element `Title` i `chunk_by_title` robił z niego samotny parent (tylko tytuł,
  bez treści). Fix w ekstraktorze wprowadzony równolegle — skrypt sprząta dane
  już zaingestowane.
