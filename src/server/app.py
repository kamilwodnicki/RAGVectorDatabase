from fastapi import FastAPI
from src.server.routes import router

app = FastAPI(title="RAG Query Server")
app.include_router(router)
