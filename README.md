SYSTEM ZARZĄDZANIA BAZĄ WEKTOROWĄ RAG
=====================================

System ładuje pliki PDF/TXT, klasyfikuje ich elementy (tytuł, akapit, lista, tabela),
dzieli na **parent/child chunki**, generuje embeddingi modelem `intfloat/multilingual-e5-base`
(dim 768) i zapisuje:

- **childy** (małe fragmenty ~400 znaków) → **Qdrant** (wyszukiwanie wektorowe)
- **parenty** (szerszy kontekst ~2000 znaków) → **MongoDB** (zwracane jako odpowiedź)
- **stan plików** (SHA-256, powiązane ID) → **MongoDB** (inkrementalna synchronizacja)

Serwer FastAPI udostępnia wyszukiwanie semantyczne — zapytanie trafia do Qdrant, a wynikiem są
odpowiadające parenty z MongoDB (szerszy kontekst niż sam fragment, który dopasował się do pytania).

---

## Przygotowanie

1. Zainstaluj Dockera.
2. Wrzuć pliki PDF/TXT do folderu `./DOKUMENTY/`.
3. Skopiuj `.env.example` do `.env` i ustaw `CUDA_VARIANT` odpowiedni dla swojej karty GPU.

---

## ⚠️ Akceleracja GPU — konfiguracja Docker

Ingest wymaga GPU. Kontenery Docker nie mają dostępu do karty graficznej bez dodatkowej konfiguracji hosta.

Typowy błąd przy braku konfiguracji:
> `Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]`

### Jak skonfigurować GPU na hoście

1. **Sterowniki** — użyj narzędzi systemowych (np. `ubuntu-drivers` na Ubuntu). Pakiety ze strony NVIDIA mogą wymusić wersję sterownika niekompatybilną ze starszą kartą.

2. **Wersja CUDA** — sprawdź `nvidia-smi` na hoście. Wersja CUDA projektu nie może przekraczać maksymalnej wspieranej przez zainstalowany sterownik.

3. **NVIDIA Container Toolkit** — jedyne oficjalne narzędzie umożliwiające dostęp GPU z kontenera Docker. Instaluje się na hoście.
   - 🔗 [Dokumentacja NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

## Uruchomienie (Docker)

```bash
make build   # Zbuduj obrazy (tylko przy pierwszym uruchomieniu)
make up      # Uruchom kontenery (rag-server + qdrant + mongodb)
make shell   # Wejdź do kontenera rag-server
make down    # Zatrzymaj kontenery
make logs    # Podgląd logów
```

---

## Komendy (wewnątrz kontenera)

### Zarządzanie bazą

```bash
# Utwórz kolekcję Qdrant (wymiar 768, metryka COSINE)
python manage.py db setup

# Usuń i utwórz kolekcję od nowa
python manage.py db setup --recreate

# Stan Qdrant (liczba childów)
python manage.py db status

# Stan MongoDB (liczba parentów)
python manage.py db mongo-status
```

### Ingest — synchronizacja dokumentów

```bash
python manage.py ingest run
```

**Domyślnie działa inkrementalnie** — dla każdego pliku w `./DOKUMENTY/` porównuje SHA-256
zawartości ze stanem zapisanym w MongoDB (`rag.files_metadata`) i wybiera jedną z akcji:

| Akcja | Warunek | Co robi |
|-------|---------|---------|
| `ADD` | Brak w metadata store | Ekstrahuje, chunkuje, zapisuje parenty i childy |
| `UPDATE` | Hash różny od zapisanego | Usuwa stare wektory/parenty, przetwarza od nowa |
| `SKIP` | Hash identyczny | Pomija całkowicie — brak ekstrakcji i embeddingu |
| `DELETE` | Plik zniknął z dysku, a jest w bazie | Usuwa wszystkie powiązane wektory i parenty |

Wymaga GPU (embedding). Błędy na jednym pliku nie przerywają operacji — plik dostaje
`status: ERROR` w `files_metadata`.

### Ingest — pełna przebudowa

```bash
python manage.py ingest rebuild
```

Kasuje **wszystko** (Qdrant collection + `rag.parents` + `rag.files_metadata`) i indeksuje
od zera. **Wymaga potwierdzenia**. Pomiń prompt flagą `--yes`.

### Strategia ekstrakcji

Domyślnie `fast` (pdfminer). Dla skanów lub złożonych układów:

```bash
python manage.py ingest run --strategy hi_res
```

### Serwer zapytań

```bash
python manage.py serve
```

Uruchamia FastAPI na `0.0.0.0:8000`.

**Zapytanie:**
```bash
POST /query/
Content-Type: application/json

{ "query": "treść pytania", "k": 3 }
```

`k` = liczba fragmentów wyszukanych w Qdrant. Zwrócone zostaną **unikalne parenty** do których
należały dopasowane childy — liczba wyników może być mniejsza od `k` (kilka childów z jednego
parenta → jeden wynik).

**Odpowiedź:**
```json
{
  "results": [
    {
      "content": "pełny fragment parenta (~2000 znaków)...",
      "metadata": { "source": "plik.pdf", "page": 5, "parent_id": "uuid" }
    }
  ]
}
```

---

## Chunking: parent/child

`chunk_by_title` z `unstructured` tworzy **parenty** na granicach sekcji (nowy `Title` = nowy
parent). Parametry w `src/config.py`:

| Parametr | Wartość | Znaczenie |
|----------|---------|-----------|
| `PARENT_MAX_SIZE` | 2000 | Hard-limit — parent nigdy nie przekracza |
| `PARENT_SOFT_SIZE` | 1500 | Po przekroczeniu zamyka na najbliższej granicy elementu |
| `PARENT_COMBINE_UNDER` | 800 | Sekcje poniżej tej wielkości łączone (bez śmieciowych mini-parentów) |
| `CHILD_CHUNK_SIZE` | 400 | Rozmiar childa embedowanego w Qdrant |
| `CHILD_CHUNK_OVERLAP` | 80 | Overlap między sąsiednimi childami w obrębie parenta |

Każdy child zawiera `parent_id` w metadanych. Tabele nie są cięte, o ile mieszczą się w `PARENT_MAX_SIZE`.

---

## Strategie ekstrakcji PDF

### `fast` (domyślna)

Analizuje natywną warstwę tekstową PDF przy użyciu `pdfminer`. Szybka, nie wymaga
dodatkowych zależności systemowych ani modeli ML.

**Używaj gdy:**
- Dokumenty mają natywną warstwę tekstową (nie są skanami)
- Prosty układ: jedna kolumna, typowy dokument biurowy lub akademicki
- Programy teatralne, regulaminy, artykuły, sprawozdania

### `hi_res`

Renderuje każdą stronę jako obraz i przepuszcza przez model computer vision
(`detectron2`), który wykrywa regiony layoutu tak jak widzi je człowiek.
Następnie OCR (tesseract) wyciąga tekst z każdego regionu osobno.

**Używaj gdy:**
- Dokument jest skanem (brak warstwy tekstowej)
- Wielokolumnowy układ (gazeta, magazyn, historyczne materiały)
- Tabele z danymi liczbowymi są ważne dla wyników wyszukiwania
- Tekst osadzony w grafikach lub plakatach

**Wymaga dodatkowej konfiguracji:**

W `Dockerfile` odkomentuj linię instalującą `poppler`:

```dockerfile
RUN apt-get install -y poppler-utils
```

Oraz dodaj do `requirements/base.txt`:

```
unstructured-inference
```

Po zmianach wykonaj `make build`.

### Języki dokumentów

Domyślnie obsługiwany jest **polski + angielski** (`EXTRACTION_LANGUAGES=pol,eng`). Nadpisz
w `.env` jeśli potrzebujesz innej konfiguracji, np.:

```
EXTRACTION_LANGUAGES=pol,eng,deu
```

Dla `fast` języki są tylko metadanymi. Dla `hi_res` trafiają do tesseracta i realnie wpływają na
jakość OCR polskich diakrytyków.

Model embeddingowy (`multilingual-e5-base`) jest wielojęzyczny — dokumenty w różnych językach
można przeszukiwać jednym zapytaniem w dowolnym z obsługiwanych języków.

### Domyślna strategia przez zmienną środowiskową

Zamiast podawać `--strategy` przy każdym ingecie:

```
EXTRACTION_STRATEGY=hi_res
```

---

## Architektura

| Moduł | Rola |
|-------|------|
| `src/config.py` | Wszystkie stałe: model, Qdrant/Mongo, parent/child params, urządzenia, języki |
| `src/db/client.py`, `collection.py` | Qdrant — klient, zarządzanie kolekcją |
| `src/db/mongo.py` | MongoDB — klient, kolekcja `parents` |
| `src/db/metadata_store.py` | Śledzenie stanu plików (`files_metadata`): hash, akcje ADD/UPDATE/SKIP/DELETE |
| `src/extractor/` | Ekstrakcja PDF/TXT przez `unstructured`, czyszczenie (NFKC, CID, hyphen-break) |
| `src/ingest/chunker.py` | `chunk_by_title` → parenty; `RecursiveCharacterTextSplitter` → childy z `parent_id` |
| `src/ingest/pipeline.py` | `run_sync` (inkrementalny) i `run_rebuild` (pełny reset) |
| `src/ingest/embeddings.py` | E5 z prefixami `passage:` / `query:` |
| `src/server/` | FastAPI: `/query/` łączy wynik z Qdrant z parentami z MongoDB |
| `src/commands/` | Komendy CLI: `db`, `ingest` |
| `manage.py` | Główny punkt wejścia CLI |

### Infrastruktura Docker

- **rag-server** — aplikacja (ingest + FastAPI), wymaga GPU
- **qdrant** — baza wektorowa, volume `qdrant_data`, porty 6333/6334
- **mongodb** — docstore parentów + metadata store, volume `mongodb_data`, port 27017, baza `rag`

---

## Konfiguracja CUDA_VARIANT

Ustaw w `.env` przed `make build`:

| Wartość | Karty | Wymagania |
|---------|-------|-----------|
| `cu118` | GTX 1060, RTX 2080, RTX 3090 | sterownik ≥ 520 |
| `cu126` | RTX 4090, RTX 5080, RTX 5090 | sterownik ≥ 560 |
| `cpu`   | brak GPU / CI | — |
