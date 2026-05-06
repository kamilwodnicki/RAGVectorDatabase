from contextlib import contextmanager
from time import perf_counter

from prometheus_client import Histogram

LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
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


@contextmanager
def observe(histogram: Histogram, **labels):
    start = perf_counter()
    try:
        yield
    finally:
        histogram.labels(**labels).observe(perf_counter() - start)
