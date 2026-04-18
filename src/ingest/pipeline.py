import typer
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, INGEST_DEVICE, PDF_SOURCE_DIR, EXTRACTION_STRATEGY
from src.db.collection import setup_collection
from src.extractor.pipeline import extract_documents
from src.ingest.splitter import get_splitter
from src.ingest.embeddings import E5HuggingFaceEmbeddings


def run_ingest(source_dir: str = PDF_SOURCE_DIR, strategy: str = EXTRACTION_STRATEGY) -> None:
    if INGEST_DEVICE is None:
        typer.secho("BŁĄD: Brak GPU (CUDA). Przerwanie operacji.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Urządzenie: {INGEST_DEVICE} | Strategia ekstrakcji: {strategy}")

    docs = extract_documents(source_dir, strategy=strategy)
    if not docs:
        typer.secho(f"BŁĄD: Brak dokumentów w {source_dir}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Wyekstrahowano {len(docs)} stron. Dzielę na chunki...")
    chunks = get_splitter().split_documents(docs)
    typer.echo(f"Utworzono {len(chunks)} chunków. Zapisuję do Qdrant...")

    setup_collection(recreate=True)

    embeds = E5HuggingFaceEmbeddings(device=INGEST_DEVICE)
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeds,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        collection_name=COLLECTION_NAME,
    )

    typer.secho(f"Gotowe — zapisano {len(chunks)} chunków w kolekcji '{COLLECTION_NAME}'.", fg=typer.colors.GREEN)
