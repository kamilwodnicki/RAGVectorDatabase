SYSTEM ZARZĄDZANIA BAZĄ WERTOROWĄ RAG
=====================================

OPIS SYSTEMU:
System pozwala na profesjonalne zarządzanie procesem Ingest (ładowania PDF), 
wyboru aktywnej bazy oraz testowania jakości (benchmark). 
Domyślnie używa modelu E5-Base (wymiar 768).

PRZYGOTOWANIE (WINDOWS & UBUNTU):
1. Upewnij się, że masz zainstalowanego Dockera.
2. Wrzuć pliki PDF do folderu: /DOKUMENTY

-----------------------------------------------------------
URUCHOMIENIE SYSTEMU (DOCKER)
-----------------------------------------------------------
UBUNTU / LINUX:
Zbudowanie kontenera (należy wykonać tylko podczas pierwszego uruchomienia):
```bash
   make build
```

Aby uruchomić kontener należy wykonać:
```bash
   make up
```

Aby wejść do kontenera należy wykonać:
```bash
   make shell
```
-----------------------------------------------------------
WINDOWS (PowerShell):
Zbudowanie kontenera (należy wykonać tylko podczas pierwszego uruchomienia):
```bash
   docker-compose build
```
Aby uruchomić kontener należy wykonać:
```bash
   docker-compose up -d
```

Aby wejść do kontenera należy wykonać:
```bash
   docker-compose run --rm rag-server /bin/bash
```
---
# KOMENDY ZARZĄDZANIA (WYKONYWANE WEWNĄTRZ KONTENERA)
Po wejściu do kontenera (shell), używaj punktu wejścia manage.py:

## 1. WYBÓR GŁÓWNEJ BAZY (Interaktywny):
```bash
   python manage.py db select
```
> (System wyświetli opcje T1-T4 oraz L1-L5, wpisz wybraną nazwę)

## 2. SPRAWDZENIE STATUSU:
```bash
   python manage.py db status
```
> Pokaże, która baza jest obecnie używana przez API

## 3. GENEROWANIE BAZY (INGEST):
   - Wszystkie warianty (20 baz):
      ```bash
      python manage.py db ingest --variant all
      ```
   - Konkretny wariant (np. Smart 800):
      ```bash
      python manage.py db ingest --type T4_Smart --variant L3_800
      ```
## 4. ANALIZA ROZMIARU:
   ```bash
      python manage.py db analyze
   ```
> Wyświetla raport zajętości miejsca na dysku

---
# TESTY

## TEST 1 — Benchmark jakości RAG (`test run`)

Mierzy **jakość wyszukiwania semantycznego** — czy system zwraca właściwe fragmenty
w odpowiedzi na pytania z zestawu testowego (`test_dataset.json`).

**Wymaga:** działającego serwera API (`manage.py serve`) uruchomionego w tle.

```bash
# Uruchom serwer w tle
python manage.py serve &

# Benchmark dla konkretnego wariantu bazy
python manage.py test run --type T4_Smart --variant L3_800

# Z własnym URL serwera
python manage.py test run --type T1_Sztywny --variant L2_500 --url http://127.0.0.1:8000/query/
```

**Jak działa:**
1. Wczytuje pytania i oczekiwane fragmenty z `test_dataset.json`
2. Wysyła każde pytanie do API (`POST /query/`)
3. Porównuje zwrócone chunki z oczekiwanym fragmentem używając modelu sędziowskiego
   (`paraphrase-multilingual-MiniLM-L12-v2`, podobieństwo cosinusowe)
4. Uznaje trafienie (hit) gdy `cosine_score >= 0.45`

**Metryki wynikowe:**
- **Hit Rate (%)** — odsetek pytań, dla których system znalazł właściwy fragment
- **Avg Semantic Score** — średnie podobieństwo cosinusowe między oczekiwanym
  fragmentem a najlepiej dopasowanym chunkiem (zakres 0–1, im wyżej tym lepiej)

**Format pliku `test_dataset.json`:**
```json
[
  {
    "question": "Treść pytania po polsku?",
    "expected_snippet": "Fragment tekstu który powinien zostać zwrócony przez bazę"
  }
]
```

---

## TEST 2 — Benchmark prędkości embeddingów (`speedtest run`)

Mierzy **szybkość modelu embeddingowego** (E5-Base) na CPU i GPU.
Nie wymaga uruchomionego serwera ani bazy wektorowej.

```bash
# Podstawowe uruchomienie (200 tekstów na test długości)
python manage.py speedtest run

# Więcej tekstów dla stabilniejszych pomiarów
python manage.py speedtest run --texts 500

# Zmiana liczby tekstów rozgrzewkowych (warmup)
python manage.py speedtest run --texts 300 --warmup 10
```

**Parametry:**

| Opcja | Domyślnie | Opis |
|-------|-----------|------|
| `--texts` / `-n` | 200 | Liczba tekstów embedowanych w teście długości (TEST 1) |
| `--warmup` / `-w` | 5 | Liczba tekstów rozgrzewkowych (eliminuje czas ładowania CUDA) |

**Co mierzy — TEST 1 (wpływ długości tekstu):**

Embeduje `N` tekstów w każdym wariancie długości i mierzy czas oraz przepustowość.
Pozwala ocenić jak długość chunka wpływa na czas przetwarzania.

| Wariant | ~Słów | Odpowiada wariantowi bazy |
|---------|-------|--------------------------|
| bardzo_krotki | 50 | chunki T1/T2 z małym L |
| krotki | 200 | L2_500 |
| sredni | 500 | L3_800 / L4_1200 |
| dlugi | 1000 | L5_1600 |

**Co mierzy — TEST 2 (wpływ liczby tekstów / rozmiaru batcha):**

Embeduje rosnące zestawy tekstów o stałej długości (~200 słów) i mierzy
przepustowość w tekstach/s oraz czas na jeden tekst (ms/tekst).
GPU zyskuje tu najbardziej, bo operacje macierzowe skalują się lepiej przy dużych batchach.

| Batch | Co modeluje |
|-------|-------------|
| 1 | pojedyncze zapytanie użytkownika przez API |
| 10–50 | wyszukiwanie w małym zestawie chunków |
| 100–500 | ingest fragmentów z kilkudziesięciu stron PDF |

**Wynik końcowy** (gdy dostępne GPU): tabela porównawcza CPU vs GPU
z kolumną `Speedup` (ile razy GPU jest szybsze od CPU).

---

# ARCHITEKTURA PLIKÓW
- src/core/config.py: Stałe (modele, typy splitterów, wymiar 768).
- src/core/engine.py: Logika przetwarzania PDF i embeddingów.
- src/core/persistence.py: Zapisywanie wyboru aktywnej bazy w JSON.
- src/commands/: Moduły komend CLI (db, test).
- manage.py: Główny kontroler CLI (odpowiednik bin/console).