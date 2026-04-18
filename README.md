SYSTEM ZARZĄDZANIA BAZĄ WEKTOROWĄ RAG
=====================================

System ładuje pliki PDF, dzieli je na chunki, generuje embeddingi modelem E5-Base (dim 768)
i zapisuje do bazy wektorowej Qdrant. Serwer FastAPI udostępnia wyszukiwanie semantyczne przez JSON.

---

## Przygotowanie

1. Zainstaluj Dockera.
2. Wrzuć pliki PDF do folderu `./DOKUMENTY/`.
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
make up      # Uruchom kontenery (rag-server + qdrant)
make shell   # Wejdź do kontenera
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

# Sprawdź stan kolekcji
python manage.py db status
```

### Ingest — załadowanie dokumentów

```bash
python manage.py ingest run
```

Ładuje wszystkie PDF i TXT z `./DOKUMENTY/`, dzieli na chunki (rozmiar 800, overlap 20%)
i indeksuje w Qdrant. Wymaga GPU.

Domyślna strategia ekstrakcji to `fast`. Aby użyć `hi_res`:

```bash
python manage.py ingest run --strategy hi_res
```

---

## Strategie ekstrakcji PDF

Ekstrakcja używa biblioteki `unstructured`, która klasyfikuje każdy blok tekstu w dokumencie
(tytuł, tekst główny, przypis, nagłówek strony itd.) i zachowuje tylko treść merytoryczną.

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
Następnie OCR wyciąga tekst z każdego regionu osobno.

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

### Domyślna strategia przez zmienną środowiskową

Zamiast podawać `--strategy` przy każdym ingecie, możesz ustawić domyślną strategię
globalnie w `.env`:

```
EXTRACTION_STRATEGY=hi_res
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

**Odpowiedź:**
```json
{
  "results": [
    { "content": "fragment dokumentu...", "metadata": { "source": "plik.pdf", "page": 5 } }
  ]
}
```

---

## Architektura

| Moduł | Rola |
|-------|------|
| `src/config.py` | Wszystkie stałe: model, Qdrant, rozmiar chunka, urządzenia |
| `src/db/` | Klient Qdrant i zarządzanie kolekcją |
| `src/ingest/` | Ładowanie PDF, chunking, embeddingi, pipeline |
| `src/server/` | FastAPI: schematy, router, aplikacja |
| `src/commands/` | Komendy CLI: `db`, `ingest` |
| `manage.py` | Główny punkt wejścia CLI |

---

## Konfiguracja CUDA_VARIANT

Ustaw w `.env` przed `make build`:

| Wartość | Karty | Wymagania |
|---------|-------|-----------|
| `cu118` | GTX 1060, RTX 2080, RTX 3090 | sterownik ≥ 520 |
| `cu126` | RTX 4090, RTX 5080, RTX 5090 | sterownik ≥ 560 |
| `cpu`   | brak GPU / CI | — |
