# src/commands/db.py
import typer
import os
import shutil
import chromadb
from src.core.config import SPLITTER_TYPES, LENGTH_VARIANTS, BASE_DB_DIR, PDF_SOURCE_DIR, INGEST_DEVICE
from src.core.engine import E5HuggingFaceEmbeddings, get_splitter, load_documents
from src.core.persistence import save_active_db, get_active_db
from langchain_chroma import Chroma

app = typer.Typer()

@app.command("select")
def select():
    """Interaktywny wybór aktywnej bazy."""
    typer.secho("\n--- DOSTĘPNE TYPY ---", fg=typer.colors.CYAN)
    for t in SPLITTER_TYPES: typer.echo(f" - {t}")
    s_type = typer.prompt("Wybierz typ")

    typer.secho("\n--- DOSTĘPNE DŁUGOŚCI ---", fg=typer.colors.CYAN)
    for l in LENGTH_VARIANTS.keys(): typer.echo(f" - {l}")
    s_len = typer.prompt("Wybierz długość")

    save_active_db(s_type, s_len)
    typer.secho(f"Ustawiono {s_type}/{s_len} jako aktywną.", fg=typer.colors.GREEN)

@app.command("ingest")
def ingest(
    variant: str = typer.Option("all", "--variant", "-v"),
    splitter_type: str = typer.Option("all", "--type", "-t")
):
    if INGEST_DEVICE is None:
        typer.secho("BŁĄD: System nie wykrył karty GPU (CUDA). Przerwanie operacji.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Uruchamiam generowanie na: {INGEST_DEVICE}")
    docs = load_documents(PDF_SOURCE_DIR)
    embeds = E5HuggingFaceEmbeddings(device=INGEST_DEVICE)
    types = SPLITTER_TYPES if splitter_type == "all" else [splitter_type]
    lengths = LENGTH_VARIANTS if variant == "all" else {variant: LENGTH_VARIANTS[variant]}

    for t_name in types:
        for l_name, size in lengths.items():
            path = os.path.join(BASE_DB_DIR, t_name, l_name)
            if os.path.exists(path): shutil.rmtree(path)
            
            splitter = get_splitter(t_name, size)
            chunks = splitter.split_documents(docs)
            
            client = chromadb.PersistentClient(path=path)
            Chroma.from_documents(documents=chunks, embedding=embeds, client=client, collection_name="docs")
            typer.echo(f"Zapisano: {t_name}/{l_name}")

@app.command("status")
def status():
    """Pokazuje aktualnie aktywną bazę danych."""
    active = get_active_db()
    typer.secho(f"Aktywna baza: {active['db_type']}/{active['db_variant']}", fg=typer.colors.GREEN)

@app.command("analyze")
def analyze():
    """Analiza rozmiaru bazy na dysku."""
    for root, dirs, files in os.walk(BASE_DB_DIR):
        if 'chroma.sqlite3' in files:
            size = sum(os.path.getsize(os.path.join(root, f)) for f in files) / (1024*1024)
            typer.echo(f"Baza: {os.path.relpath(root, BASE_DB_DIR)} | Rozmiar: {size:.2f} MB")