# RAG Vector Database

System RAG (Retrieval-Augmented Generation) dla dokumentów wielojęzycznych — domyślnie polskich i angielskich, konfigurowalnie także innych języków wspieranych przez `intfloat/multilingual-e5-base`. Ładuje pliki PDF/TXT, dzieli je na **parent/child chunki**, embeduje children modelem E5 (dim 768) i wystawia FastAPI do wyszukiwania semantycznego zwracającego szerszy kontekst parenta. Dokumenty w różnych językach mogą leżeć w jednym korpusie i być przeszukiwane jednym zapytaniem.

## Spis treści

- [Czym jest ten projekt](#czym-jest-ten-projekt)
- [Architektura danych](#architektura-danych)
- [Wymagania](#wymagania)
- [Konfiguracja GPU](#konfiguracja-gpu)
- [Szybki start](#szybki-start)
- [Komendy Make](#komendy-make)
- [Komendy CLI](#komendy-cli)
- [API](#api)
- [Konfiguracja (`.env`)](#konfiguracja-env)
- [Tryby wyszukiwania](#tryby-wyszukiwania)
- [Proponowane ustawienia](#proponowane-ustawienia)
- [Strategie ekstrakcji PDF](#strategie-ekstrakcji-pdf)
- [Tuning — co zmienić kiedy](#tuning--co-zmienić-kiedy)
- [Logi](#logi)
- [Testy](#testy)
- [Deploy na produkcję](#deploy-na-produkcję)
- [Struktura projektu](#struktura-projektu)
- [Troubleshooting](#troubleshooting)

---

## Czym jest ten projekt

**Co robi:**

- Inkrementalnie indeksuje katalog dokumentów (PDF/TXT) — wykrywa pliki nowe, zmienione i usunięte po hashu SHA-256
- Dzieli dokumenty na dwupoziomowe chunki (parent/child) i przechowuje w dwóch bazach:
  - **Qdrant** — wektory dense (E5) + sparse (BM25) dla każdego małego child chunka (~400 znaków)
  - **MongoDB** — pełne teksty parentów (~2000 znaków) + stan plików (`files_metadata`)
- Wystawia HTTP API (`POST /query/`), które wyszukuje dopasowane children w Qdrant i zwraca odpowiadające parenty jako wynik
- Wspiera trzy tryby wyszukiwania: **dense** (semantyczny), **sparse** (leksykalny BM25), **hybrid** (weighted RRF fusion)

**Czym nie jest:** to nie jest generator odpowiedzi LLM. API zwraca surowe fragmenty dokumentów — integracja z LLM (np. jako kontekst dla promptu) to zadanie warstwy wyżej.

**Dla kogo:** wielojęzyczne korpusy dokumentów (domyślnie PL/EN, rozszerzalne przez `EXTRACTION_LANGUAGES`), gdzie istotne są zarówno zapytania konwersacyjne ("jak działa X?"), jak i szukanie po rzadkich terminach (akronimy, kody, nazwy własne). Model embeddingowy jest wielojęzyczny — pytanie po polsku może trafić w dokument angielski i odwrotnie.

---

## Architektura danych

```
Dokumenty w ./DOKUMENTY/  (.pdf | .txt | .json)
           ↓
    extractor (per-extension):
      .pdf  → unstructured.partition_pdf (fast | hi_res)
      .txt  → unstructured.partition_text
      .json → article loader (id/date/title/content)
           ↓
    cleaner: NFKC, (cid:NNNN) removal, hyphen-break fix, whitespace
           ↓
  chunker:  chunk_by_title → parenty (~2000 znaków)
            RecursiveCharacterTextSplitter → children (~400 znaków)
           ↓
  ┌────────────────────────────┬────────────────────────────┐
  │  Qdrant (children)         │  MongoDB (parenty)         │
  │  - dense vector (E5, 768d) │  - rag.parents             │
  │  - sparse vector (BM25)    │  - rag.files_metadata      │
  │  - payload: source,        │    (hash, powiązane ID)    │
  │    filename, file_ext,     │                            │
  │    page, ingested_at,      │                            │
  │    parent_id, text,        │                            │
  │    article_id/date/title   │                            │
  └────────────────────────────┴────────────────────────────┘
           ↓
    POST /query/ →  retrieval (dense | sparse | hybrid)
           ↓
    Dedupe parent_id → fetch parents z MongoDB → zwrot JSON
```

**Dlaczego parent/child:** mały chunk trafnie dopasowuje się do zapytania (precision), duży rodzic daje LLM-owi dostateczny kontekst (recall). Każdy child nosi `parent_id` w payloadzie, po dopasowaniu jest rozwiązywany do parenta.

**Named vectors w Qdrant:** jeden punkt trzyma jednocześnie wektor dense i sparse. Dzięki temu przełączanie trybu wyszukiwania nie wymaga przebudowy bazy.

### Obsługiwane formaty źródłowe

| Rozszerzenie | Ekstraktor | Uwagi |
|---|---|---|
| `.pdf` | `unstructured.partition_pdf` | strategia `fast` (pdfminer) lub `hi_res` (OCR + layout) |
| `.txt` | `unstructured.partition_text` | zwykły tekst |
| `.json` | `extract_json_article` | artykuł w schemacie `{id, date, title, content}` |

**Format `.json`:**

```json
{
  "id": "52",
  "date": "2004-01-09",
  "title": "Teatry walczą o ustawę",
  "content": "Pełen tekst artykułu..."
}
```

- `content` pusty (`""`) → plik **pomijany** (zapisywany hash, kolejne `ingest run` nie próbuje powtórnie)
- `id`, `date`, `title` → trafiają do payloadu Qdranta jako `article_id`, `article_date`, `article_title` (z indeksami) i do MongoDB parents → można po nich filtrować w `/query/` i widzisz je w odpowiedzi
- pole `date` traktowane jako `datetime` w Qdrant payload — wspiera range queries (`{"article_date": {"gte": "2020-01-01"}}`)

---

## Wymagania

| Komponent | Wersja | Uwagi |
|-----------|--------|-------|
| Docker | 20.10+ | Wymagany Compose V2 (`docker compose`, nie `docker-compose`) |
| Make | dowolna | Wrapper na docker compose |
| GPU NVIDIA | RTX 20xx / 30xx / 40xx / 50xx | Wymagane **tylko do ingestu**. API działa na CPU. |
| Sterownik NVIDIA | zgodny z wybranym `CUDA_VARIANT` | Patrz tabela [CUDA_VARIANT](#konfiguracja-cuda_variant) |
| NVIDIA Container Toolkit | najnowsza | Jedyne wsparte narzędzie dostępu GPU z Dockera |

Porty: `8000` (API), `6333`/`6334` (Qdrant REST/gRPC), `27017` (MongoDB). Jeśli któryś jest zajęty — zmień mapowanie w `docker-compose.yml`.

---

## Konfiguracja GPU

Ingest wymaga CUDA (skrypt aborduje jeśli `torch.cuda.is_available()` zwraca `False`). Kontenery Docker nie mają dostępu do karty bez osobnej konfiguracji.

**Typowy błąd przy braku konfiguracji:**
> `Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]`

### Kroki

1. **Sterownik** — zainstaluj przez narzędzia systemowe (`ubuntu-drivers autoinstall` na Ubuntu). Pakiety ze strony NVIDIA potrafią wymusić wersję niekompatybilną ze starszymi kartami.

2. **Sprawdź wersję CUDA w sterowniku** — `nvidia-smi`, pole "CUDA Version" w nagłówku. To **sufit** runtime: nie możesz użyć wariantu PyTorcha wyższego niż to pole.

3. **NVIDIA Container Toolkit** — instalacja na hoście. [Oficjalna dokumentacja](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

4. **Test** — po wszystkim: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi` powinien pokazać kartę.

### Konfiguracja `CUDA_VARIANT`

Wybór wariantu wheeli PyTorcha. Ustaw w `.env` **przed** `make build`:

| Wartość | Karty (typowe) | Wymaga sterownika |
|---------|----------------|-------------------|
| `cu118` | GTX 1060, RTX 2080, RTX 3090 | ≥ 520 |
| `cu126` | RTX 4090, RTX 5080, RTX 5090 | ≥ 560 |
| `cpu`   | brak GPU / CI / lokalny dev bez ingestu | — |

Zmiana `CUDA_VARIANT` wymaga `make build` (nowe wheele w obrazie).

---

## Szybki start

```bash
# 1. Konfiguracja
cp .env.example .env
# Otwórz .env i ustaw CUDA_VARIANT zgodnie ze swoją kartą.

# 2. Wrzuć dokumenty (.pdf, .txt lub .json — patrz "Obsługiwane formaty źródłowe")
cp twoje_dokumenty/*.pdf ./DOKUMENTY/

# 3. Build + start (pierwszy raz trwa dłużej — pobiera obrazy, buduje)
make build
make up

# 4. Wejdź do kontenera i zaindeksuj
make shell
python manage.py ingest run   # wewnątrz kontenera

# 5. Uruchom serwer (nadal w kontenerze)
python manage.py serve
```

Test z hosta:

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Czym jest przetrenowanie modelu?", "k": 3}'
```

---

## Komendy Make

Wszystkie komendy uruchamiane **z katalogu projektu na hoście**.

| Komenda | Co robi |
|---------|---------|
| `make build` | Buduje obraz `rag-server`. Uruchom przy pierwszym użyciu i po zmianach `Dockerfile`/`requirements/`/`CUDA_VARIANT`. |
| `make up` | Startuje wszystkie serwisy (`rag-server`, `qdrant`, `mongodb`) w tle. |
| `make down` | Zatrzymuje i usuwa kontenery. Volumeny (`qdrant_data`, `mongodb_data`) **zostają** — dane przetrwają. |
| `make shell` | Wchodzi do `rag-server` jako `/bin/bash`. Tu odpalasz `python manage.py ...`. |
| `make logs` | Śledzi logi `rag-server` (`Ctrl+C` żeby wyjść). |
| `make test-unit` | Pytest z markerem `unit` — szybkie, bez zewnętrznych usług. |
| `make test-integration` | Pytest `integration` — wymaga działających Qdrant + MongoDB (czyli `make up`). |
| `make test-eval` | Pytest `eval` — evaluacja jakości retrievalu na golden secie (Hit@k, MRR). Wymaga GPU. |
| `make prepare_pr` | Uruchamia po kolei: unit → integration → eval. **Pierwszy fail zatrzymuje cały łańcuch** (`pytest -x` + zależności `make`). |

Pełne wipe'owanie danych (np. przy zmianie schematu):

```bash
make down
docker volume rm rag_qdrant_data rag_mongodb_data
make up
```

---

## Komendy CLI

Uruchamiane **wewnątrz kontenera** (`make shell` najpierw). Entrypoint: `manage.py` (Typer).

### `db` — zarządzanie kolekcjami

```bash
python manage.py db setup              # Tworzy kolekcję Qdrant (hybrid: dense + sparse) + payload indexes
python manage.py db setup --recreate   # Usuwa i tworzy od nowa
python manage.py db status             # Liczba chunków, status kolekcji
python manage.py db mongo-status       # Liczba parentów w MongoDB
```

`db setup` jest idempotentne — wywołanie na istniejącej poprawnej kolekcji tylko dopiania brakujące indeksy. Jeśli kolekcja ma **stary schemat** (bez named vectors), zgłasza `CollectionSchemaMismatchError` i wymaga `ingest rebuild`.

### `ingest` — synchronizacja dokumentów

```bash
# Inkrementalna (domyślna) — tylko zmiany
python manage.py ingest run

# Inny folder źródłowy
python manage.py ingest run --source ./DOKUMENTY

# Inna strategia ekstrakcji (skany, złożone układy)
python manage.py ingest run --strategy hi_res

# Pełna przebudowa — kasuje wszystko, reindex od zera (wymaga potwierdzenia)
python manage.py ingest rebuild
python manage.py ingest rebuild --yes      # pomija prompt
```

**Akcje inkrementalnego ingestu** (decyzja per plik, na podstawie hash SHA-256):

| Akcja | Warunek | Efekt |
|-------|---------|-------|
| `ADD` | Brak w `files_metadata` | Ekstrakcja → chunking → embedding → zapis |
| `UPDATE` | Hash różny | Usunięcie starych wektorów/parentów → przetworzenie od nowa |
| `SKIP` | Hash identyczny | Nic — natychmiast |
| `DELETE` | Plik zniknął z dysku, jest w metadata | Usunięcie powiązanych wektorów i parentów |

Błąd na jednym pliku nie przerywa operacji — plik dostaje `status: ERROR` w `files_metadata`. Pozostałe pliki są przetwarzane normalnie.

**Kiedy `ingest rebuild`:**
- Zmiana modelu embeddingowego lub wymiaru wektora
- Zmiana schematu kolekcji (migracja dense-only → hybrid)
- Dodanie nowych pól metadata, które muszą być obecne na wszystkich punktach
- Podejrzenie niespójności między Qdrant/Mongo/metadata store

#### `ingest file <ścieżka>` — wymuszony reprocess pojedynczego pliku

```bash
# Przeprocesuj jeden plik (strategia z .env)
python manage.py ingest file ./DOKUMENTY/raport.pdf

# Z konkretną strategią — np. ten skan przez hi_res, reszta korpusu może iść na fast
python manage.py ingest file ./DOKUMENTY/skan.pdf --strategy hi_res
```

**Co robi:**
- Wymusza pełen reprocess **niezależnie od hash** — nawet jeśli plik się nie zmienił
- Jeśli plik był wcześniej w bazie → kasuje stare children (Qdrant) + parents (Mongo) → indeksuje od nowa
- Jeśli plik był nowy → po prostu indeksuje (zachowuje się jak ADD)
- Inne pliki w bazie pozostają nietknięte
- Pojedynczy plik = brak konfirmacji (operacja tania, niska blast radius)

**Use cases:**
- "Ten dokument wyszedł krzywo, chcę spróbować z `--strategy hi_res`"
- "Plik jest w stanie ERROR po crashu — retry bez touchowania"
- "Dla testów chcę zmieniać chunking jednego pliku, nie reindexując całego korpusu"
- "Mam mieszany korpus — większość PDF-ów natywnych chcę puścić przez `fast`, a 2-3 skany przez `hi_res`"

**Błędy:**
- Plik nie istnieje → `BŁĄD: Plik nie istnieje lub nie jest plikiem: ...` (exit 1)
- Nieobsługiwane rozszerzenie → `BŁĄD: Nieobsługiwane rozszerzenie '.docx'. Dozwolone: .pdf, .txt, .json`
- Brak GPU → `BŁĄD: Ingest wymaga GPU (CUDA), ale jest niedostępne.`

### `serve` — API

```bash
python manage.py serve
python manage.py serve --port 9000 --host 127.0.0.1
```

Przy starcie drukuje efektywną konfigurację (`format_effective_config()`) — przydatne do debugu.

---

## API

### `POST /query/`

**Request:**

```json
{
  "query": "treść pytania",
  "k": 3,
  "filters": {
    "filename": "umowa_2024.pdf"
  }
}
```

- `query` — treść pytania (wymagane)
- `k` — liczba children pobieranych z Qdrant (opcjonalne, default: `DEFAULT_K` = 3). Liczba zwróconych parentów może być ≤ `k` (kilka children z tego samego parenta → jeden wynik).
- `filters` — filtry po metadanych (opcjonalne, patrz niżej)

#### Filtry po metadanych

Pole `filters` zawęża kandydatów **przed** liczeniem podobieństwa — filtrowanie odbywa się po stronie Qdrant na indeksach payloadowych, więc jest szybkie nawet na dużych zbiorach.

**Dozwolone pola:** `source`, `filename`, `file_extension`, `page`, `ingested_at`, `parent_id`, `article_id`, `article_date`, `article_title`. Pole `text` nie jest dozwolone (i nie miałoby sensu — treść to przecież to, co dopasowuje `query`).

**Warianty wartości:**

| Forma | Znaczenie | Przykład |
|-------|-----------|----------|
| Skalar | Równość (`MatchValue`) | `{"filename": "x.pdf"}` |
| Lista | OR — dowolna z wartości (`MatchAny`) | `{"filename": ["a.pdf", "b.pdf"]}` |
| Dict z operatorami | Zakres (`Range` dla liczb, `DatetimeRange` dla ISO 8601) | `{"page": {"gte": 5, "lte": 20}}` |

**Dozwolone operatory zakresów:** `gte`, `lte`, `gt`, `lt`.

Wszystkie warunki łączone **AND** (Qdrant `must`).

**Przykłady:**

```json
{ "query": "...", "filters": {"file_extension": "pdf"} }

{ "query": "...", "filters": {"filename": ["raport_q1.pdf", "raport_q2.pdf"]} }

{ "query": "...", "filters": {"page": {"gte": 10, "lte": 50}} }

{ "query": "...", "filters": {
    "ingested_at": {"gte": "2026-01-01T00:00:00+00:00"},
    "file_extension": "pdf"
}}

{ "query": "reforma teatrów", "filters": {
    "article_date": {"gte": "2000-01-01", "lte": "2010-12-31"}
}}

{ "query": "...", "filters": {"article_id": ["52", "108", "203"]} }
```

**Błędy:**
- Pole spoza whitelisty → `HTTP 400` z komunikatem `"Pole 'X' nie jest dozwolone..."`
- Nieznany operator zakresu → `HTTP 400`
- Filtr nie zwracający żadnych wyników → `HTTP 200` z pustą listą `{"results": []}`

**Uwaga o starych danych:** filtry po `filename`/`file_extension`/`ingested_at` działają tylko na punktach zaingestowanych po wprowadzeniu tych pól. Starsze punkty nie mają tych pól w payloadzie i nie będą spełniać warunku. Jeśli chcesz spójnych wyników — `ingest rebuild`.

**Response:**

```json
{
  "results": [
    {
      "content": "pełny tekst parenta (~2000 znaków)...",
      "metadata": {
        "source": "/app/DOKUMENTY/dokument.pdf",
        "filename": "dokument.pdf",
        "file_extension": "pdf",
        "page": 5,
        "ingested_at": "2026-04-23T14:32:17+00:00",
        "parent_id": "a3f1b2c4-...",
        "article_id": null,
        "article_date": null,
        "article_title": null
      }
    }
  ]
}
```

Pusta lista `[]` gdy korpus jest pusty lub żaden parent nie pasuje.

### Przykład curl

Proste pytanie:

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "jak działa architektura parent-child?", "k": 5}'
```

Z filtrami:

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "warunki umowy",
    "k": 3,
    "filters": {
      "file_extension": "pdf",
      "page": {"gte": 1, "lte": 10}
    }
  }'
```

### `POST /ingest/sync/`

Synchronizacja korpusu — odpowiednik CLI `ingest run` (bez `paths`) lub `ingest file` (z `paths`). Jeden endpoint dwa zachowania, dobierane na podstawie obecności pola `paths`.

**Request:**

```json
{
  "paths": ["DOKUMENTY/raport.pdf", "DOKUMENTY/skan.pdf"],
  "strategy": "hi_res"
}
```

- `paths` — opcjonalne. **Brak / pusta lista** → pełen incremental sync całego `DOKUMENTY/` (akcje ADD/UPDATE/SKIP/DELETE jak w `ingest run`). **Z listą** → reprocess **tylko podanych plików**, każdy wymuszony niezależnie od hash (jak `ingest file <path>` w pętli). Bez SKIP/DELETE w drugim trybie.
- `strategy` — opcjonalna, default `EXTRACTION_STRATEGY` z `.env`. Wartości: `fast` \| `hi_res`. Override globalnej strategii dla tego wywołania.

**Defense-in-depth:** ścieżki w `paths` MUSZĄ być wewnątrz `PDF_SOURCE_DIR` (`DOKUMENTY/`). Pierwsza ścieżka spoza katalogu → cały request odrzucony jako `HTTP 400`. Klient nie może triggerować odczytu dowolnych plików z systemu.

**Response (200 OK):**

```json
{
  "added": 3,
  "updated": 1,
  "skipped": 18,
  "deleted": 0,
  "errors": [],
  "elapsed_seconds": 87.4,
  "strategy": "hi_res"
}
```

| Pole | Znaczenie |
|------|-----------|
| `added` | Pliki nowe (nie były w metadata store) |
| `updated` | Pliki które już były w bazie i zostały reprocessed |
| `skipped` | Pliki niezmienione (tylko w trybie full sync — bez `paths`) |
| `deleted` | Pliki usunięte z dysku (tylko w trybie full sync) |
| `errors` | Lista `{"path": ..., "error": ...}` dla plików które padły. Operacja nie przerywa się przy błędzie pojedynczego pliku. |
| `elapsed_seconds` | Czas wykonania całego syncu (do orientacji w wydajności) |
| `strategy` | Strategia faktycznie użyta |

**Kody błędów:**

| Kod | Kiedy |
|-----|-------|
| `400` | Ścieżka w `paths` spoza `DOKUMENTY/` |
| `503` | Brak GPU (`Ingest wymaga GPU (CUDA)...`) lub awaria Mongo (metadata store) |
| `500` | Niespodziewany błąd globalny (traceback w `./logs/app.log`) |

Pojedyncze pliki które padły są w `errors`, **nie** powodują 5xx. Pomyłka per-plik to normalna sytuacja, nie awaria całego endpointa.

**Przykłady curl:**

```bash
# Tryb 1: pełen incremental sync (jak ingest run)
curl -X POST http://localhost:8000/ingest/sync/ \
  -H "Content-Type: application/json" \
  -d '{}'

# Tryb 2: konkretne pliki, force reprocess (jak ingest file <path> w pętli)
curl -X POST http://localhost:8000/ingest/sync/ \
  -H "Content-Type: application/json" \
  -d '{"paths": ["DOKUMENTY/raport_q1.pdf", "DOKUMENTY/raport_q2.pdf"]}'

# Mix: konkretne pliki + override strategii
curl -X POST http://localhost:8000/ingest/sync/ \
  -H "Content-Type: application/json" \
  -d '{"paths": ["DOKUMENTY/skan.pdf"], "strategy": "hi_res"}'
```

### `POST /ingest/rebuild/`

Pełna przebudowa — kasuje **wszystko** (Qdrant collection, `rag.parents`, `rag.files_metadata`) i indeksuje od zera. Odpowiednik CLI `ingest rebuild --yes`.

**Request:**

```json
{
  "confirm": "DELETE_ALL",
  "strategy": "fast"
}
```

- `confirm` — **wymagane**, MUSI równać się literałowi `"DELETE_ALL"`. Inaczej `HTTP 400`. Defense-in-depth: zapobiega przypadkowym wywołaniom (np. typo w URL `/ingest/sync/` → `/ingest/rebuild/`).
- `strategy` — opcjonalna, jak w `/ingest/sync/`.

**Response:** identyczny schemat jak `/ingest/sync/` — zwraca `IngestSyncResponse` ze statystykami sync wykonanego po wyczyszczeniu baz. Po rebuildzie wszystkie pliki będą w `added` (nigdy `updated`/`skipped`/`deleted`, bo baza jest pusta przed reindeksacją).

**Kody błędów:**

| Kod | Kiedy |
|-----|-------|
| `400` | Brak / zły `confirm` (`Wymagane potwierdzenie: pole "confirm"...`) |
| `503` | Brak GPU lub awaria Mongo |
| `500` | Niespodziewany błąd |

**Przykład curl:**

```bash
curl -X POST http://localhost:8000/ingest/rebuild/ \
  -H "Content-Type: application/json" \
  -d '{"confirm": "DELETE_ALL"}'
```

### Uwaga o latencji i bezpieczeństwie endpointów ingest

**Synchroniczność:** oba endpointy są **synchroniczne**. Mogą trwać minuty/godziny dla dużych korpusów lub `hi_res`. Klient HTTP musi mieć odpowiedni timeout. **Jeśli klient się rozłączy w trakcie — proces na serwerze nadal się dokończy**, dane wylądują w bazie, a wynik znajdziesz w `./logs/app.log` (szukaj `FILE-DONE`, `Synchronizacja zakończona`).

**Bezpieczeństwo:** oba endpointy są **otwarte** (jak `/query/`) — w wdrożeniu produkcyjnym wystaw je tylko za firewallem / VPN-em / proxy z auth. Wywołują kosztowne operacje (GPU, pisanie i kasowanie z baz), więc nie powinny być publicznie dostępne.

### OpenAPI / Swagger

FastAPI wystawia dokumentację pod `http://localhost:8000/docs`.

---

## Konfiguracja (`.env`)

Skopiuj `.env.example` → `.env` i zmień tylko to, co ma być inne niż default. Wszystkie zmienne mają sensowne wartości domyślne — większość projektów nie wymaga zmian poza `CUDA_VARIANT`.

### Build

| Zmienna | Default | Opis |
|---------|---------|------|
| `CUDA_VARIANT` | `cu118` | Wariant wheeli PyTorcha. `cu118` \| `cu126` \| `cpu`. Zmiana wymaga `make build`. |

### Ekstrakcja

| Zmienna | Default | Opis |
|---------|---------|------|
| `EXTRACTION_STRATEGY` | `fast` | `fast` (pdfminer) \| `hi_res` (OCR + layout detection) |
| `EXTRACTION_LANGUAGES` | `pol,eng` | Lista ISO 639-2/3 po przecinku. Dla `hi_res` przekazywane do tesseracta. |

### Chunking — parenty (MongoDB)

| Zmienna | Default | Opis |
|---------|---------|------|
| `PARENT_MAX_SIZE` | `2000` | Twardy limit znaków na parent. Nigdy nie przekraczany. |
| `PARENT_SOFT_SIZE` | `1500` | Po przekroczeniu chunker zamyka parent na najbliższej granicy elementu. Trzymaj ~75% `PARENT_MAX_SIZE`. |
| `PARENT_COMBINE_UNDER` | `800` | Sekcje krótsze są scalane — zapobiega mini-parentom. |
| `PARENT_OVERLAP` | `0` | Overlap między sąsiednimi parentami. `0` = brak. |

### Chunking — children (Qdrant)

| Zmienna | Default | Opis |
|---------|---------|------|
| `CHILD_CHUNK_SIZE` | `400` | Rozmiar child chunka w znakach. Mniejsze = precyzja, większe = kontekst. Typowo 300–600. |
| `CHILD_CHUNK_OVERLAP` | `80` | Overlap między sąsiadami. Typowo 15–25% `CHILD_CHUNK_SIZE`. |

### Retrieval

| Zmienna | Default | Opis |
|---------|---------|------|
| `DEFAULT_K` | `3` | Liczba children pobieranych z Qdrant, gdy klient nie poda `k`. |
| `RETRIEVAL_MODE` | `hybrid` | `dense` \| `sparse` \| `hybrid`. Patrz [Tryby wyszukiwania](#tryby-wyszukiwania). |

### Hybrid tuning (używane gdy `RETRIEVAL_MODE=hybrid`)

| Zmienna | Default | Opis |
|---------|---------|------|
| `HYBRID_DENSE_WEIGHT` | `1.0` | Waga kanału dense w weighted RRF. |
| `HYBRID_SPARSE_WEIGHT` | `1.0` | Waga kanału sparse w weighted RRF. |
| `HYBRID_RRF_K` | `60` | Stała RRF. Tłumi wpływ niskich pozycji rankingu. Rzadko się zmienia. |
| `SPARSE_MODEL_NAME` | `Qdrant/bm25` | Model sparse embeddings (fastembed). |

Formuła RRF: `score(d) = w_dense / (K + rank_d) + w_sparse / (K + rank_s)`

### Infrastruktura (zwykle nie ruszać)

| Zmienna | Default | Opis |
|---------|---------|------|
| `QDRANT_HOST` | `qdrant` (w Dockerze) / `localhost` (lokalnie) | Host Qdranta. |
| `QDRANT_PORT` | `6333` | Port REST. |
| `COLLECTION_NAME` | `documents` | Nazwa kolekcji Qdrant. Testy nadpisują na `documents_test`/`documents_eval`. |
| `MONGODB_HOST` | `mongodb` / `localhost` | Host MongoDB. |
| `MONGODB_PORT` | `27017` | Port MongoDB. |
| `MONGODB_DB` | `rag` | Baza Mongo. |

---

## Tryby wyszukiwania

Każdy child punkt w Qdrant ma **oba wektory** (dense + sparse) — przełączanie trybu to decyzja query-time, nie wymaga przebudowy bazy.

### `dense` — E5 multilingual (neuronowy)

- Model: `intfloat/multilingual-e5-base` (768d)
- **Dobre dla:** parafraz, synonimów, zapytań konwersacyjnych ("jak działa X?"), różnic językowych (pytanie PL, dokument ENG)
- **Słabsze dla:** rzadkich akronimów, nazw własnych, kodów technicznych których model nie widział w treningu

### `sparse` — BM25 (leksykalny, nie-neuronowy)

- Model: `Qdrant/bm25` (fastembed)
- **Dobre dla:** rzadkich terminów, kodów (np. `trf-03`, `LoRA`, `SKU-2025-A`), dokładnych dopasowań
- **Słabsze dla:** parafraz, synonimów — traktuje formy słów niemal dosłownie

### `hybrid` — weighted RRF fusion (default)

Pobiera wyniki z obu kanałów (3×`k` kandydatów z każdego, min. 30), łączy reciprocal rank fusion z konfigurowalnymi wagami. Dokument wysoko w obu rankingach dostaje boost, który trudno zbić jednym kanałem.

- **Pokrywa oba przypadki**, kompromis cenowy: dwa zapytania do Qdrant zamiast jednego
- **Domyślny** — w praktyce najbezpieczniejszy wybór startowy

---

## Proponowane ustawienia

### Start — "po prostu działa"

```bash
CUDA_VARIANT=cu118              # lub cu126 w zależności od karty
EXTRACTION_STRATEGY=fast
RETRIEVAL_MODE=hybrid
# reszta defaultowa
```

Nie zmieniaj chunking/hybrid weightów dopóki nie zobaczysz konkretnego problemu w metrykach (Hit@k) lub odpowiedziach.

### Dużo akronimów / kodów / nazw własnych

```bash
HYBRID_SPARSE_WEIGHT=2.0        # podbij kanał leksykalny
HYBRID_DENSE_WEIGHT=1.0
```

### Głównie konwersacyjne pytania w języku naturalnym

```bash
HYBRID_DENSE_WEIGHT=2.0
HYBRID_SPARSE_WEIGHT=1.0
```

### Skany / wielokolumnowe PDF-y / tabele kluczowe

```bash
EXTRACTION_STRATEGY=hi_res
```

Wymaga doinstalowania `poppler-utils`, `tesseract-ocr` + paczek językowych w `Dockerfile` oraz `unstructured-inference` w `requirements/base.txt`. Pełna instrukcja krok po kroku w [Strategie ekstrakcji PDF](#strategie-ekstrakcji-pdf).

### Krótkie pytania wymagające precyzji

```bash
CHILD_CHUNK_SIZE=300            # mniejsze = bardziej skoncentrowane dopasowania
CHILD_CHUNK_OVERLAP=60
```

### Długie pytania wymagające szerokiego kontekstu

```bash
CHILD_CHUNK_SIZE=500
PARENT_MAX_SIZE=2500
PARENT_SOFT_SIZE=2000
```

---

## Strategie ekstrakcji PDF

### `fast` (default)

`pdfminer` czyta natywną warstwę tekstową. Szybka, bez modeli ML, bez dodatkowych zależności.

**Używaj dla:** dokumentów z warstwą tekstową, jednokolumnowych, typowych biurowych/akademickich.

### `hi_res`

Renderuje każdą stronę jako obraz → `detectron2` wykrywa regiony layoutu (akapit, tabela, nagłówek) → tesseract robi OCR regionów.

**Używaj dla:** skanów, wielokolumnowych (gazety, magazyny), dokumentów gdzie tabele z liczbami są kluczowe, tekstu w grafikach.

**Czego potrzebuje (3 etapy ekstrakcji = 3 zestawy zależności):**

| Etap | Zależność | Gdzie instalować |
|------|-----------|------------------|
| PDF → obrazy stron | `poppler-utils` | system (apt) — w `Dockerfile` |
| Wykrycie layoutu | `unstructured-inference` | Python — w `requirements/base.txt` |
| OCR regionów | `tesseract-ocr` + paczki językowe (`tesseract-ocr-pol`, `tesseract-ocr-eng`, …) | system (apt) — w `Dockerfile` |

**Mapowanie kodów ISO → nazwa pakietu apt** (paczki językowe muszą zgadzać się z `EXTRACTION_LANGUAGES`):

| Kod ISO | Pakiet apt |
|---------|-----------|
| `pol` | `tesseract-ocr-pol` |
| `eng` | `tesseract-ocr-eng` |
| `deu` | `tesseract-ocr-deu` |
| `ukr` | `tesseract-ocr-ukr` |
| `fra` | `tesseract-ocr-fra` |
| `…`   | `tesseract-ocr-<kod>` |

#### Krok po kroku — przełączenie z `fast` na `hi_res`

**Krok 1.** W `Dockerfile` rozszerz blok `apt-get install` o `poppler-utils` i tesseracta. Pełna sekcja powinna wyglądać tak (dodajesz 4 ostatnie linie przed `&& rm -rf ...`, paczki językowe dopasuj do swojego `EXTRACTION_LANGUAGES`):

```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-pol \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
```

**Krok 2.** W `requirements/base.txt` dopisz na końcu:

```
unstructured-inference
```

**Krok 3.** Z hosta (poza kontenerem) przebuduj i uruchom obraz:

```bash
make down
make build
make up
```

`make down` jest istotne — bez niego `make up` używałby starego kontenera. Sam `make build` tworzy nowy obraz, ale dopiero `make up` go uruchamia.

**Krok 4.** Włącz `hi_res`. Dwie opcje:

A) **Per-ingest** (jednorazowo, bez zmian w `.env`):

```bash
make shell
python manage.py ingest run --strategy hi_res
```

B) **Domyślnie dla wszystkich ingestów** — w `.env`:

```
EXTRACTION_STRATEGY=hi_res
```

Po zmianie `.env`: `make down && make up` (zmienne env są wstrzykiwane przy starcie kontenera).

**Krok 5.** Weryfikacja przed pierwszym ingestem:

```bash
make shell
tesseract --list-langs
```

Powinno pokazać twoje języki (np. `pol`, `eng`) plus `osd` (default tesseracta). Jeśli któregoś brakuje — dopisz odpowiedni `tesseract-ocr-<kod>` do `Dockerfile` i powtórz Krok 3.

#### Najczęstsze błędy

| Błąd | Przyczyna | Fix |
|------|-----------|-----|
| `tesseract is not installed or its not in your PATH` | Brak systemowego binarza `tesseract-ocr` | Krok 1 (paczka `tesseract-ocr` w apt) + Krok 3 |
| `Error opening data file ... pol.traineddata` | Brak paczki językowej | Dodaj `tesseract-ocr-pol` do `Dockerfile` + Krok 3 |
| `pdf2image.exceptions.PDFInfoNotInstalledError` | Brak `poppler-utils` | Dodaj `poppler-utils` do `Dockerfile` + Krok 3 |
| `ModuleNotFoundError: No module named 'unstructured_inference'` | Brak Pythonowego pakietu | Dodaj `unstructured-inference` do `requirements/base.txt` + Krok 3 |
| Wszystko wygląda dobrze, ale ingest super wolny | `hi_res` jest z natury 10–50× wolniejszy od `fast` | To normalne. Używaj `hi_res` tylko dla dokumentów które tego wymagają (skany, wielokolumnowe). Jeśli wciąż za wolno — sprawdź sekcję "Weryfikacja GPU" niżej. |

#### Weryfikacja GPU dla layout detection (KRYTYCZNE dla wydajności)

`hi_res` ma trzy etapy per strona: render PDF → bitmapa, **layout detection (detectron2)**, OCR (tesseract). Etap środkowy to model CV który **musi** lecieć na GPU — na CPU jest **10–50× wolniejszy**. Jeśli detectron2 cicho się cofnie do CPU, masz 100 stron za 30–90 minut zamiast 3–5 minut.

Najgorsze: nikt Cię o tym nie poinformuje. Ingest "po prostu działa" — tylko bardzo, bardzo wolno.

**Diagnoza w 30 sekund:**

```bash
make shell
python -c "
import torch
print('torch CUDA available:', torch.cuda.is_available())
print('torch CUDA version :', torch.version.cuda)
from unstructured_inference.models.base import get_model
m = get_model('yolox')
device = next(m.model.parameters()).device
print('detectron2 device  :', device)
"
```

**Co powinieneś zobaczyć (poprawnie):**

```
torch CUDA available: True
torch CUDA version : 11.8
detectron2 device  : cuda:0
```

**Co oznacza, że jest źle:**

| Output | Diagnoza | Fix |
|--------|----------|-----|
| `torch CUDA available: False` + `torch CUDA version: None` | Masz CPU-only wariant PyTorcha | Sprawdź `CUDA_VARIANT` w `.env` — jeśli `cpu`, zmień na `cu118` lub `cu126`. Po zmianie: `make down && make build && make up`. |
| `torch CUDA available: True` ale `detectron2 device: cpu` | Torch widzi GPU, ale detectron2 z jakiegoś powodu poszedł na CPU | Restart kontenera (`make down && make up`). Jeśli nie pomoże — sprawdź `nvidia-smi` na hoście, czy karta nie jest zajęta przez inny proces, który blokuje CUDA. |
| Komenda wywala `ModuleNotFoundError: unstructured_inference` | Pakiet niedoinstalowany | Patrz Krok 2 wyżej (dodaj `unstructured-inference` do `requirements/base.txt`). |
| `nvidia-smi` na hoście nie widzi karty | Sterownik / NVIDIA Container Toolkit nie skonfigurowany | Patrz [Konfiguracja GPU](#konfiguracja-gpu). |

**Empiryczna kontrola czasu (test na kilku stronach):**

```bash
make shell
time python -c "
from unstructured.partition.pdf import partition_pdf
elements = partition_pdf('DOKUMENTY/twoj_plik.pdf', strategy='hi_res', languages=['pol'])
print(f'{len(elements)} elementów')
"
```

Orientacyjnie:
- **GPU OK**: ~2–5 sekund per strona
- **CPU silent fallback**: ~30–90 sekund per strona

Jeśli masz drugi wariant — wracaj do diagnostyki wyżej.

### Języki

`EXTRACTION_LANGUAGES=pol,eng` — dla `fast` to metadane, dla `hi_res` realnie wpływa na jakość OCR (polskie diakrytyki). Model embeddingowy jest wielojęzyczny — w jednym korpusie mogą być dokumenty w różnych językach.

---

## Tuning — co zmienić kiedy

Diagnozuj przez `make test-eval` — golden set pokazuje Hit@1/3/5, MRR i per-query ranking.

| Symptom | Najpierw | Jeśli nie pomoże |
|---------|----------|------------------|
| Dobre pytania lądują w rank 4-10 | Zwiększ `k` w requescie API | Tuning hybrid weightów |
| Rzadkie terminy / kody źle dopasowane | `HYBRID_SPARSE_WEIGHT=2.0` | Zwiększ `CHILD_CHUNK_SIZE` (więcej kontekstu) |
| Parafrazy nie dopasowują się | `HYBRID_DENSE_WEIGHT=2.0` | Sprawdź czy pytanie i dokument są w tym samym języku |
| Odpowiedź obcięta, brak kontekstu | Zwiększ `PARENT_MAX_SIZE`/`PARENT_SOFT_SIZE` | Zmień strategię retrievalu na LLM (reranker) |
| Za dużo małych parentów | Zwiększ `PARENT_COMBINE_UNDER` | Sprawdź strukturę dokumentu (Title heurystyka) |
| Ingest mieli godzinami | Sprawdź że `INGEST_DEVICE=cuda` | Przejdź na `fast` strategy jeśli jest `hi_res` |

Po każdej zmianie wag / chunkingu rób `ingest rebuild` lub przynajmniej `db setup --recreate`, żeby evaluacja była na świeżych danych.

---

## Logi

Aplikacja loguje równolegle do **dwóch miejsc**:

| Cel | Co | Przeżywa restart kontenera? |
|-----|-----|----------------------------|
| `./logs/app.log` (na hoście) | Plik tekstowy z **codzienną rotacją** o północy. Archiwa: `app.log.YYYY-MM-DD`. Domyślnie 14 dni historii. | Tak — to volume mount |
| stdout kontenera | Ten sam strumień; `make logs` / `docker logs rag-server` | Tylko dopóki kontener istnieje |

**Po crashu zaglądaj do `./logs/app.log`** — `make logs` pokaże pustkę bo procesu już nie ma, a plik zostanie. Plik kończy się na ostatnim wpisie tuż przed śmiercią procesu, w tym pełnym tracebackiem niezłapanych wyjątków (`sys.excepthook` przechwytuje je przed wyjściem Pythona).

### Format

```
2026-04-25 14:32:17 [INFO] src.ingest.pipeline: ADD DOKUMENTY/raport.pdf (12 parents / 348 children)
2026-04-25 14:32:18 [ERROR] src.ingest.pipeline: ERR-ADD DOKUMENTY/skan.pdf: tesseract is not installed
Traceback (most recent call last):
  File "/app/src/ingest/pipeline.py", line 189, in run_sync
  ...
```

### Konfiguracja (`.env`)

| Zmienna | Default | Opis |
|---------|---------|------|
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |
| `LOG_DIR` | `/app/logs` | Katalog wewnątrz kontenera (zmapowany na `./logs` na hoście) |

Logi nie są automatycznie kasowane przez aplikację. Bieżący plik to `app.log`, archiwa dzienne: `app.log.YYYY-MM-DD`. Cykl życia logów = cykl życia release katalogu (workflow `deploy.yml` usuwa releases starsze niż `KEEP_RELEASES`, razem z ich logami).

Zmiana `LOG_LEVEL=DEBUG` da wglądnie w bardzo szczegółowe logi (przydatne przy debug, ale plik puchnie szybciej). Po edycji `.env` → `make down && make up`.

### Przydatne komendy

```bash
# Co się stało ostatnio
tail -100 ./logs/app.log

# Śledzenie na żywo (alternatywa dla make logs)
tail -f ./logs/app.log

# Tylko błędy
grep ERROR ./logs/app.log

# Wszystkie pliki logu (po rotacji — bieżący + archiwa dzienne)
ls -lh ./logs/
# Przykład: app.log  app.log.2026-04-24  app.log.2026-04-23  ...

# Co się działo wczoraj
less ./logs/app.log.$(date -d yesterday +%Y-%m-%d)
```

### Co jest logowane

- **Każdy ingest pojedynczego pliku** (`ADD`/`UPD`/`DEL`/`SKIP`) z liczbą parentów/children
- **Każde zapytanie do API** (`/query/`) z `k`, `mode`, filtrami, początkiem treści query (200 znaków)
- **Każdy błąd** z `exc_info=True` (pełny traceback)
- **Niezłapane wyjątki** przed śmiercią procesu (poziom `CRITICAL`)
- **Konfiguracja loggera** przy starcie (potwierdza że logowanie zostało uruchomione)

### Czego NIE jest logowane

- Pełne treści zwracanych dokumentów (za duże, niepotrzebne dla debugowania)
- Pełne payloady requestów do Qdranta
- Embeddingi
- Sekrety z `.env` (typer/uvicorn nie wypisują ich)

---

## Testy

Trzy warstwy z markerami pytest:

| Warstwa | Marker | Czas | Zależności |
|---------|--------|------|-----------|
| Unit | `unit` | ~5 s | Brak — mocki, czyste funkcje |
| Integration | `integration` | ~1-2 min | Qdrant + MongoDB (`make up`) + GPU |
| Eval | `eval` | ~20 s | Qdrant + MongoDB + GPU, golden set |

```bash
make test-unit           # najtańsze, uruchamiaj często
make test-integration    # przed commitem
make test-eval           # przed zmianą modelu / strategii retrievalu
make prepare_pr          # wszystko po kolei, fail-fast
```

`pytest.ini` ma `-x` w `addopts` — **pierwszy falujący test zatrzymuje suitę**. Między suitami zatrzymuje już `make` (niezerowy kod wyjścia kończy łańcuch).

**Eval — golden set:** `tests/eval/fixtures/golden_set.json` zawiera ~20 pytań z oczekiwanym źródłem. Test drukuje rank każdego pytania i zbiorcze Hit@1, Hit@3, Hit@5, MRR@10. Używaj jako feedback loop przy tuningu.

Testy integracyjne i eval używają **osobnych kolekcji** (`documents_test`, `documents_eval`) i baz (`rag_test`, `rag_eval`) — nie konfliktują z produkcyjną bazą.

---

## Deploy na produkcję

Wdrożenia są **ręczne, wersjonowane przez tagi git, oparte o release directories** z atomowym przepięciem symlinka. Każdy deploy to konkretny tag (np. `v0.1.0`) — niezmienialna referencja do commita, który poszedł na produkcję.

### Architektura deploymentu na serwerze

```
$DEPLOY_PATH/
├── current → releases/v0.1.0_20260505_180000      # symlink do aktywnej wersji
├── releases/
│   ├── v0.0.5_20260501_120000/                    # poprzednie wersje (do rollbacku)
│   ├── v0.1.0_20260505_180000/                    # bieżąca wersja
│   │   ├── src/, docker-compose.yml, Dockerfile, ...
│   │   ├── logs/                                  # logi tego release (lokalne)
│   │   ├── DOKUMENTY → ../../shared/DOKUMENTY     # symlink do trwałych danych
│   │   └── .env      → ../../shared/.env
│   └── ...
└── shared/
    ├── DOKUMENTY/        # korpus dokumentów (trwałe)
    └── .env              # sekrety i konfiguracja (trwałe)
```

`releases/` trzyma 5 ostatnich wersji (`KEEP_RELEASES` w `deploy.yml`). Starsze są usuwane razem z ich logami. Named volumes Dockera (`rag_qdrant_data`, `rag_mongodb_data`) są niezależne od tej struktury — przeżywają wszystkie deploye.

### Wymagania jednorazowe

#### Po stronie serwera

1. **Użytkownik deploy** (np. `gh-deployer`) musi:
   - mieć skonfigurowane `~/.ssh/authorized_keys` z publicznym kluczem deploya
   - być w grupie `docker`: `sudo usermod -aG docker gh-deployer && sudo pkill -KILL -u gh-deployer`
   - mieć prawa zapisu do `$DEPLOY_PATH`
2. **Struktura katalogów**:
   ```bash
   sudo mkdir -p $DEPLOY_PATH/{releases,shared/DOKUMENTY}
   sudo touch $DEPLOY_PATH/shared/.env  # uzupełnij realną konfiguracją
   sudo chown -R gh-deployer:gh-deployer $DEPLOY_PATH
   ```

#### Po stronie GitHuba — sekrety repozytorium

| Sekret | Wartość |
|--------|---------|
| `DEPLOY_HOST` | adres serwera (np. `rag-serwer.instytut`) |
| `DEPLOY_USER` | użytkownik SSH (`gh-deployer`) |
| `DEPLOY_PORT` | port SSH (zazwyczaj `22`) |
| `DEPLOY_SSH_KEY` | **prywatny** klucz SSH (cała zawartość, łącznie z `-----BEGIN/END-----`) |
| `DEPLOY_PATH` | absolutna ścieżka root deploymentu na serwerze (bez końcowego `/`) |

#### Po stronie GitHuba — branch protection na `main`

Settings → Branches → Add rule dla `main`:
- ✅ Require a pull request before merging
- ✅ Require status checks to pass — wybierz `unit-tests`
- ✅ Restrict who can push to matching branches

### Cykl wydania — krok po kroku

1. **Praca rozwojowa**: branch z fixem/feature → PR do `main` → CI musi przejść zielono → merge przez squash.
2. **Decyzja o wydaniu**: lokalnie utwórz tag wskazujący na konkretny commit `main` i wypchnij:

   ```bash
   git checkout main
   git pull origin main
   git tag -a v0.1.0 -m "Krótki opis tej wersji"
   git push origin v0.1.0
   ```

   Konwencja: [SemVer](https://semver.org) — `MAJOR.MINOR.PATCH`. Patche (`v0.1.0` → `v0.1.1`) dla fixów, minor (`v0.1.0` → `v0.2.0`) dla nowych ficzerów, major (`v0.1.0` → `v1.0.0`) dla zmian łamiących kompatybilność API.

3. **Sprawdź na GitHubie, że CI dla tego commita zakończyło się sukcesem** (Actions → ostatni run na `main` przy commicie, na który wskazuje tag).

4. **Odpal deploy**: Actions → **Manual Remote Deploy** → **Run workflow**:
   - Branch: `main` (nie ma znaczenia, decyduje `tag`)
   - **Tag**: `v0.1.0`
   - Run workflow

   Workflow:
   1. Pobiera kod z taga.
   2. Weryfikuje, że `unit-tests` przeszły zielono dla commita pod tagiem (fail-fast jeśli nie).
   3. Tworzy `releases/v0.1.0_<timestamp>/` na serwerze.
   4. rsynchronizuje kod (z exclude'ami: `.git`, `tests`, `DOKUMENTY`, `.env` itd.).
   5. Linkuje `shared/*` do release.
   6. Atomowo przepina symlink `current`.
   7. `docker compose up -d --build`.
   8. Czyści releases starsze niż 5 ostatnich.

5. **Sanity check po deployu**:

   ```bash
   ssh gh-deployer@rag-serwer
   cd $DEPLOY_PATH/current
   docker compose ps                                       # 3 kontenery Up
   docker exec rag-server python manage.py db status       # Qdrant odpowiada
   docker exec rag-server python manage.py db mongo-status # Mongo OK
   ```

### Rollback

Wszystkie poprzednie wersje siedzą w `releases/`. Cofnięcie to **podmiana symlinka** + restart:

```bash
ssh gh-deployer@rag-serwer
cd $DEPLOY_PATH

# Sprawdź co masz dostępne:
ls -1t releases/

# Przepnij na konkretną starą wersję (przykład):
ln -sfn releases/v0.0.5_20260501_120000 current

# Wstaw stack na starym kodzie:
cd current && docker compose up -d
```

Bez rebuildu, w sekundy. **Dane (DOKUMENTY, MongoDB, Qdrant) pozostają nietknięte** — rollback dotyczy wyłącznie kodu.

### Ograniczenia bieżącej implementacji

- **Brak automatycznego health-checka** po `docker compose up`. Jeśli kontener wstanie, ale aplikacja w środku zaraz padnie, workflow zaraportuje sukces. Sanity check (`docker compose ps` + curl) trzeba zrobić ręcznie po deployu.
- **Migracje danych** (zmiany schematu MongoDB / Qdrant) nie są częścią workflow — jeśli przyszły release tego wymaga, trzeba to obsłużyć osobno przed lub po deployu.
- **Sekrety w `.env`** są wspólne dla wszystkich releases — zmiana wymaga ręcznego edytu `$DEPLOY_PATH/shared/.env` na serwerze, niezależnie od kodu.

---

## Struktura projektu

```
.
├── DOKUMENTY/                  # Wejście: .pdf / .txt / .json (volume w Dockerze)
├── src/
│   ├── config.py               # Wszystkie zmienne env + format_effective_config()
│   ├── commands/               # CLI (db, ingest)
│   ├── extractor/              # partition_pdf / partition_text / json_article + cleaner
│   ├── ingest/
│   │   ├── chunker.py          # chunk_by_title → parenty, RecursiveSplitter → children
│   │   ├── embeddings.py       # E5 z prefixami passage:/query:
│   │   ├── sparse_embeddings.py# BM25 wrapper (fastembed)
│   │   └── pipeline.py         # run_sync (inkrementalny), run_rebuild (pełny)
│   ├── retrieval/
│   │   ├── hybrid.py           # dense / sparse / hybrid + weighted RRF
│   │   └── filters.py          # dict → qdrant Filter (MatchValue / MatchAny / Range / DatetimeRange)
│   ├── db/
│   │   ├── client.py           # Qdrant client factory
│   │   ├── collection.py       # setup_collection + payload indexes + schema guard
│   │   ├── mongo.py            # Mongo client, parents collection
│   │   └── metadata_store.py   # Single source of truth dla stanu plików (hash, powiązania)
│   └── server/
│       ├── app.py              # FastAPI instance
│       ├── routes.py           # POST /query/
│       └── schemas.py          # Pydantic: QueryRequest, DocumentFragment, QueryResponse
├── tests/
│   ├── unit/                   # Szybkie, izolowane
│   ├── integration/            # Realny Qdrant + Mongo
│   └── eval/                   # Retrieval quality na golden set
├── requirements/
│   ├── base.txt                # Wspólne dla wszystkich wariantów
│   ├── cu118.txt               # PyTorch CUDA 11.8
│   ├── cu126.txt               # PyTorch CUDA 12.6
│   └── cpu.txt                 # PyTorch CPU-only
├── Dockerfile
├── docker-compose.yml          # rag-server + qdrant + mongodb
├── Makefile
├── manage.py                   # Typer entrypoint
├── pytest.ini
├── .env.example
└── README.md
```

---

## Troubleshooting

### `could not select device driver "nvidia"`

Brak NVIDIA Container Toolkit na hoście. Patrz [Konfiguracja GPU](#konfiguracja-gpu).

### `BŁĄD: Brak GPU (CUDA). Przerwanie operacji.`

Ingest wymaga GPU. Sprawdź:
1. `nvidia-smi` na hoście — karta widoczna?
2. `docker exec rag-server python -c "import torch; print(torch.cuda.is_available())"` — powinno być `True`
3. `CUDA_VARIANT` w `.env` pasuje do wersji sterownika? (patrz tabela wyżej)
4. Po zmianie `.env` zrobiłeś `make build`? Zmiana wariantu wymaga przebudowania obrazu.

### `CollectionSchemaMismatchError`

Istniejąca kolekcja Qdrant ma stary schemat (bez named vectors). Zrób:

```bash
python manage.py ingest rebuild --yes
```

### Zmiana `.env` nie działa

- Zmienne build-time (`CUDA_VARIANT`) — wymagają `make build`
- Zmienne runtime (reszta) — wymagają `make down && make up`, bo są wstrzykiwane przy starcie kontenera
- Weryfikacja: `python manage.py serve` drukuje efektywną konfigurację na starcie

### Port 8000/6333/27017 zajęty

Zmień mapowanie w `docker-compose.yml` (lewa strona dwukropka to port hosta):

```yaml
ports:
  - "8001:8000"   # teraz API jest na http://localhost:8001
```

### Qdrant/Mongo mają dziwne dane z poprzednich eksperymentów

```bash
make down
docker volume rm rag_qdrant_data rag_mongodb_data
make up
python manage.py ingest run   # świeży index
```

### `make shell` zawiesza się / wymaga TTY

`make shell` używa `docker exec -it`. W CI / skryptach non-interactive użyj bez flag:

```bash
docker exec rag-server python manage.py db status
```

### Embedding jest wolny

Sprawdź w logach ingestu `Efektywna konfiguracja` — `Urządzenia: ingest=cuda api=cpu`. Jeśli `ingest=None` — GPU nie jest widoczne. Jeśli jest `cuda` a nadal wolno — `EXTRACTION_STRATEGY=hi_res` jest drogi, użyj `fast` jeśli dokumenty mają warstwę tekstową.
