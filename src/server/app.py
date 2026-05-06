from time import perf_counter

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.server.metrics import http_request_duration_seconds
from src.server.routes import router

app = FastAPI(title="RAG Query Server")


@app.middleware("http")
async def record_request_duration(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code),
    ).observe(perf_counter() - start)
    return response


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)
