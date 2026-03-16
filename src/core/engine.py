import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from .config import MODEL_NAME

class E5HuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, device: str, *args, **kwargs):
        kwargs.setdefault('model_kwargs', {'device': device})
        kwargs.setdefault('encode_kwargs', {'normalize_embeddings': True})
        super().__init__(model_name=MODEL_NAME, *args, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")

def get_splitter(type_id: str, chunk_size: int):
    if type_id == "T1_Sztywny":
        return RecursiveCharacterTextSplitter(separators=[" "], chunk_size=chunk_size, chunk_overlap=0, keep_separator=True)
    elif type_id == "T2_Zdania":
        return RecursiveCharacterTextSplitter(separators=[". ", "? ", "! ", "\n"], chunk_size=chunk_size, chunk_overlap=0, keep_separator=True)
    elif type_id == "T3_ZdaniaContext":
        return RecursiveCharacterTextSplitter(separators=[". ", "? ", "! ", "\n"], chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.25), keep_separator=True)
    elif type_id == "T4_Smart":
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.20), add_start_index=True)
    return None

def load_documents(source_dir: str):
    loader = PyPDFDirectoryLoader(source_dir)
    return loader.load()