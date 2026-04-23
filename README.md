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
- [Testy](#testy)
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
Dokumenty w ./DOKUMENTY/
           ↓
    unstructured.partition (fast | hi_res)
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
  │    parent_id, text         │                            │
  └────────────────────────────┴────────────────────────────┘
           ↓
    POST /query/ →  retrieval (dense | sparse | hybrid)
           ↓
    Dedupe parent_id → fetch parents z MongoDB → zwrot JSON
```

**Dlaczego parent/child:** mały chunk trafnie dopasowuje się do zapytania (precision), duży rodzic daje LLM-owi dostateczny kontekst (recall). Każdy child nosi `parent_id` w payloadzie, po dopasowaniu jest rozwiązywany do parenta.

**Named vectors w Qdrant:** jeden punkt trzyma jednocześnie wektor dense i sparse. Dzięki temu przełączanie trybu wyszukiwania nie wymaga przebudowy bazy.

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

# 2. Wrzuć dokumenty
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

**Dozwolone pola:** `source`, `filename`, `file_extension`, `page`, `ingested_at`, `parent_id`. Pole `text` nie jest dozwolone (i nie miałoby sensu — treść to przecież to, co dopasowuje `query`).

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
        "parent_id": "a3f1b2c4-..."
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

Wymaga `poppler-utils` w `Dockerfile` i `unstructured-inference` w `requirements/base.txt` (patrz [Strategie ekstrakcji](#strategie-ekstrakcji-pdf)).

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

**Wymaga:**
1. W `Dockerfile` odkomentuj `poppler-utils`:
   ```dockerfile
   RUN apt-get install -y poppler-utils
   ```
2. Dodaj do `requirements/base.txt`:
   ```
   unstructured-inference
   ```
3. `make build`

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

## Struktura projektu

```
.
├── DOKUMENTY/                  # Wejście: PDF/TXT do zaindeksowania (volume w Dockerze)
├── src/
│   ├── config.py               # Wszystkie zmienne env + format_effective_config()
│   ├── commands/               # CLI (db, ingest)
│   ├── extractor/              # partition_pdf/partition_text + cleaner (NFKC, CID, hyphen)
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
