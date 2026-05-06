# Monitoring stack — quick reference

Stack: Prometheus (metryki, retencja 60d) + Grafana (wizualizacja) + Loki (logi, retencja 60d) + Promtail (kolektor logów) + eksportery (Node, DCGM, cAdvisor, MongoDB) + natywne `/metrics` z Qdranta i FastAPI.

## Adresy

| Usługa | URL | Login |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus UI | http://localhost:9090 | – |
| Loki API | http://localhost:3100 | – |
| Node Exporter | http://localhost:9100/metrics | – |
| DCGM Exporter | http://localhost:9400/metrics | – |
| cAdvisor | http://localhost:8080 | – |
| MongoDB Exporter | http://localhost:9216/metrics | – |
| Qdrant /metrics | http://localhost:6333/metrics | – |
| FastAPI /metrics | http://localhost:8000/metrics | – |

## Dashboardy

W Grafanie: **Dashboards** w lewym menu.

- **Host** — CPU/RAM/dysk/temp/PSI/swap. Zmienna `$device` u góry — wybierz dysk z dropdowna.
- **GPU** — utylizacja, VRAM, temp (z thresholdami 75/83°C), pobór mocy.
- **Containers** — CPU/RAM/I-O/uptime/OOM per kontener. Zmienna `$container` (multi-select).
- **Databases** — Qdrant (RAM, pending writes, page faults) + Mongo (połączenia, latencje, RAM, ops).
- **Application** — `/query/` i wewnętrzne wywołania:
  - histogramy p50/p95/p99 dla `/query/`, Qdrant, Mongo
  - latency breakdown total vs Qdrant vs Mongo
  - liczba dzieci/parentów per query, top score
  - panel logów na dole z pełnymi JSON-ami zapytań

## Logi /query/ — gotowe LogQL

W **Explore → Loki → Code mode**. Wszystkie pola JSON są dostępne po `| json`.

### Podstawowe

```logql
{container="rag-server"} | json
```

```logql
{container="rag-server"} | json | event="query"
```

### Filtrowanie po trybie / modelu / eksperymencie

```logql
{container="rag-server"} | json | mode="dense"
```

```logql
{container="rag-server"} | json | mode="hybrid"
```

```logql
{container="rag-server"} | json | experiment_id="baseline-e5-k3"
```

### Anomalie / debugging

```logql
# zapytania bez wyników (n_parents = 0)
{container="rag-server"} | json | event="query" | n_parents=0
```

```logql
# zapytania ze słabym matchem (top_score < 0.7)
{container="rag-server"} | json | event="query" | top_score < 0.7
```

```logql
# wolne zapytania (>500ms)
{container="rag-server"} | json | event="query" | duration_ms > 500
```

### Czytelny one-liner per query

```logql
{container="rag-server", event="query"} | json
  | line_format "QUERY: {{.query}} | mode={{.mode}} | top={{.top_score}} | n_parents={{.n_parents}} | exp={{.experiment_id}} | {{.duration_ms}}ms"
```

### Pełny widok: kompaktowy stream + szczegóły na klik

Otwórz **Explore → Loki → Code mode** i wklej:

```logql
{container="rag-server", event="query"} | json | json children="children", parents="returned_parent_ids" | line_format "QUERY: {{.query}} | mode={{.mode}} | top={{.top_score}} | n_parents={{.n_parents}} | exp={{.experiment_id}} | {{.duration_ms}}ms"
```

Każdy log to **krótki one-liner**. Klik w wpis → "Log details" pokazuje wszystkie pola JSON-a (w tym `children` i `parents` jako tablice).

**Dlaczego dwa `| json`?**
- pierwszy `| json` (bez argumentów) — wypakowuje wszystkie skalary top-level (query, top_score, …),
- drugi `| json children="...", parents="..."` — dodatkowo wyciąga tablice (które inaczej są pomijane).

**Toggle nad logami:**
- `Wrap lines: off` — długie linie idą poziomo z scrollem zamiast się zawijać.
- `Prettify JSON: off` — wyłącz, bo psuje kompaktowy widok line_format.

**Uwaga**: `| json` bez argumentów wypakowuje tylko top-level skalary do filtracji. Tablice (`children`, `returned_parent_ids`) musisz jawnie podać po nazwie: `json children="children", parents="returned_parent_ids"`.

### Zapisanie zapytania jako Starred

Po uruchomieniu zapytania pojawia się ono w **Query history** (lewy panel w Explore — ikona zegara). Kliknij gwiazdkę → trafia do **Starred** i odtworzysz je jednym klikiem przy następnym otwarciu Explore.

### Tylko query + scores

```logql
{container="rag-server"} | json | event="query"
  | line_format "{{.query}}\n  scores: {{.children}}"
```

### Wszystkie logi (nie tylko query) z konkretnego kontenera

```logql
{container="qdrant"}
{container="mongodb"}
{container="prometheus"}
```

## Tipy dla pracy magisterskiej

### Eksperymenty z modelami / parametrami

1. Ustaw nazwę eksperymentu w `.env`:
   ```
   EXPERIMENT_ID=baseline-e5-k3-dense
   ```
2. Zrestartuj `rag-server`:
   ```bash
   docker compose up -d --force-recreate rag-server
   ```
3. Uruchom serię testów (curl, swój skrypt, itp.).
4. Zmień config (model, chunk_size, retrieval_mode...), zaktualizuj `EXPERIMENT_ID`, restart.
5. W Grafanie filtruj po `experiment_id` żeby porównać.

Każdy log JSON automatycznie zawiera `model_name`, `child_chunk_size`, `child_chunk_overlap`, `parent_max_size`, `extraction_strategy`, więc zawsze wiesz **co** było aktywne.

### Porównywanie eksperymentów

W panelu **Application → Top score (p50/p95) per mode** widzisz różnicę między dense i hybrid od razu.

W Loki Explore — zapytanie z dwoma filtrami obok siebie:
```
{container="rag-server"} | json | experiment_id=~"baseline-.*|new-model-.*"
```

### Zapisywanie queries w Grafanie

Każde zapytanie w Explore możesz zapisać:
- **"Add to dashboard"** → tworzy nowy panel z Twoim queryem,
- **"Query history"** (lewy panel w Explore) → ostatnie zapytania, można nazwać i przypiąć.

## Stack — start / stop

```bash
# Uruchom wszystko
docker compose up -d

# Status wszystkich usług
docker compose ps

# Sprawdzenie wszystkich targetów Prometheusa (powinno: 7x "up")
curl -s http://localhost:9090/api/v1/targets | grep -oE '"health":"[^"]+"' | sort | uniq -c

# Zatrzymaj
docker compose down

# Pełny restart (dane w wolumenach zostają)
docker compose down && docker compose up -d
```

## Wymagania hosta

- **Docker** w trybie `overlay2` (NIE `containerd-snapshotter`) — inaczej cAdvisor się rozsypuje. W `/etc/docker/daemon.json` musi być:
  ```json
  { "features": { "containerd-snapshotter": false } }
  ```
- **NVIDIA Container Toolkit** dla DCGM Exporter.
- Linux kernel ≥ 4.20 z włączonym **PSI** (Pressure Stall Information) — domyślnie tak na Ubuntu 22.04+.
- Dla `hwmon` (temperatury): chipy `coretemp` (Intel) lub `k10temp` (AMD). Jeśli inny, zmień regex `--collector.hwmon.chip-include` w `docker-compose.yml` (usługa `node-exporter`).

## Wolumeny (dane trwałe)

- `rag_qdrant_data` — wektory
- `rag_mongodb_data` — parents + metadata
- `rag_prometheus_data` — metryki (60 dni)
- `rag_loki_data` — logi (60 dni)
- `rag_grafana_data` — ustawienia użytkownika Grafany (dashboardy provisionowane są w repo, nie tu)

`docker compose down` zachowuje wszystkie wolumeny. Aby wymazać:
```bash
docker compose down -v
docker volume rm rag_qdrant_data rag_mongodb_data rag_prometheus_data rag_loki_data rag_grafana_data
```
