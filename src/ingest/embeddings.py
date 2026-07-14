from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import EMBEDDING_PROMPT_STYLE, MODEL_NAME, QWEN3_INSTRUCTION


class RAGEmbeddings(HuggingFaceEmbeddings):
    """Wrapper na HuggingFaceEmbeddings, dobierający styl promptu z configu.

    Style:
      e5    — prefix `passage:` / `query:` (intfloat/e5* family)
      qwen3 — instruction-tuned: dokument raw, zapytanie z `Instruct: ...\nQuery: ...`
      mmlw  — sdadas/mmlw-*: dokument raw, zapytanie z prefiksem `zapytanie: `
      none  — surowy tekst (BGE-M3, większość pozostałych)
    """

    def __init__(self, device: str, *args, **kwargs):
        kwargs.setdefault("model_kwargs", {"device": device})
        kwargs.setdefault("encode_kwargs", {"normalize_embeddings": True})
        super().__init__(model_name=MODEL_NAME, *args, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if EMBEDDING_PROMPT_STYLE == "e5":
            return super().embed_documents([f"passage: {t}" for t in texts])
        # qwen3 / mmlw / none — dokumenty bez prefixu
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if EMBEDDING_PROMPT_STYLE == "e5":
            return super().embed_query(f"query: {text}")
        if EMBEDDING_PROMPT_STYLE == "qwen3":
            return super().embed_query(f"Instruct: {QWEN3_INSTRUCTION}\nQuery: {text}")
        if EMBEDDING_PROMPT_STYLE == "mmlw":
            return super().embed_query(f"zapytanie: {text}")
        return super().embed_query(text)


# Backward-compat alias dla istniejących importów
E5HuggingFaceEmbeddings = RAGEmbeddings
