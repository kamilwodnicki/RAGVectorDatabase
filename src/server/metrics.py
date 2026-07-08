from contextlib import contextmanager
from time import perf_counter

from prometheus_client import Histogram

LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "Czas obsługi żądania HTTP od wejścia do FastAPI po zwrot odpowiedzi.",
    labelnames=("method", "endpoint", "status"),
    buckets=LATENCY_BUCKETS,
)

qdrant_query_duration_seconds = Histogram(
    "qdrant_query_duration_seconds",
    "Czas zapytania do Qdranta z perspektywy klienta Pythona.",
    labelnames=("operation",),
    buckets=LATENCY_BUCKETS,
)

mongo_query_duration_seconds = Histogram(
    "mongo_query_duration_seconds",
    "Czas zapytania do MongoDB z perspektywy klienta Pythona.",
    labelnames=("operation",),
    buckets=LATENCY_BUCKETS,
)

qdrant_children_per_query = Histogram(
    "qdrant_children_per_query",
    "Liczba child chunków zwróconych przez Qdrant per /query/.",
    labelnames=("mode",),
    buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100),
)

query_parents_returned = Histogram(
    "query_parents_returned",
    "Liczba unikalnych parentów zwróconych użytkownikowi per /query/.",
    labelnames=("mode",),
    buckets=(0, 1, 2, 3, 5, 10, 20),
)

qdrant_top_score = Histogram(
    "qdrant_top_score",
    "Score najlepszego matchu z Qdranta per /query/.",
    labelnames=("mode",),
    buckets=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
)

reranker_duration_seconds = Histogram(
    "reranker_duration_seconds",
    "Czas rerankingu kandydatów (cross-encoder) per /query/.",
    # Reranking na CPU (cross-encoder, ~20 kandydatów) potrafi trwać
    # kilkanaście–kilkadziesiąt sekund, stąd progi aż do 30 s.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
)

reranker_candidates_count = Histogram(
    "reranker_candidates_count",
    "Liczba kandydatów przekazanych do rerankera per /query/.",
    buckets=(0, 5, 10, 20, 50, 100),
)


@contextmanager
def observe(histogram: Histogram, **labels):
    start = perf_counter()
    try:
        yield
    finally:
        target = histogram.labels(**labels) if labels else histogram
        target.observe(perf_counter() - start)
