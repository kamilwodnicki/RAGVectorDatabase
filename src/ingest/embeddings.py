from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import MODEL_NAME


class E5HuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, device: str, *args, **kwargs):
        kwargs.setdefault("model_kwargs", {"device": device})
        kwargs.setdefault("encode_kwargs", {"normalize_embeddings": True})
        super().__init__(model_name=MODEL_NAME, *args, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")
