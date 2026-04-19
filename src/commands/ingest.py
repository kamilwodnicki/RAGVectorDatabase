import typer

from src.config import (
    COLLECTION_NAME,
    EXTRACTION_STRATEGY,
    MONGODB_DB,
    MONGODB_FILES_METADATA_COLLECTION,
    MONGODB_PARENTS_COLLECTION,
    PDF_SOURCE_DIR,
)
from src.ingest.pipeline import run_rebuild, run_sync

app = typer.Typer()


@app.command("run")
def run(
    source_dir: str = typer.Option(PDF_SOURCE_DIR, "--source", "-s", help="Folder z dokumentami"),
    strategy: str = typer.Option(EXTRACTION_STRATEGY, "--strategy", help="Strategia ekstrakcji PDF: fast | hi_res"),
):
    """Inkrementalna synchronizacja — dodaje nowe, aktualizuje zmienione, usuwa skasowane."""
    run_sync(source_dir=source_dir, strategy=strategy)


@app.command("rebuild")
def rebuild(
    source_dir: str = typer.Option(PDF_SOURCE_DIR, "--source", "-s", help="Folder z dokumentami"),
    strategy: str = typer.Option(EXTRACTION_STRATEGY, "--strategy", help="Strategia ekstrakcji PDF: fast | hi_res"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Pomiń potwierdzenie (użyj z rozwagą)"),
):
    """Pełne przebudowanie bazy — kasuje wszystko i indeksuje od nowa."""
    typer.secho("⚠️  UWAGA: operacja nieodwracalna", fg=typer.colors.YELLOW, bold=True)
    typer.echo("Zostaną usunięte:")
    typer.echo(f"  • Qdrant — cała kolekcja '{COLLECTION_NAME}' (wszystkie wektory)")
    typer.echo(f"  • MongoDB — '{MONGODB_DB}.{MONGODB_PARENTS_COLLECTION}' (parent documents)")
    typer.echo(f"  • MongoDB — '{MONGODB_DB}.{MONGODB_FILES_METADATA_COLLECTION}' (stan plików)")
    typer.echo(f"Następnie wszystkie pliki z '{source_dir}' zostaną zaindeksowane od zera.")
    typer.echo("")

    if not yes and not typer.confirm("Kontynuować?", default=False):
        typer.secho("Anulowano.", fg=typer.colors.YELLOW)
        raise typer.Abort()

    run_rebuild(source_dir=source_dir, strategy=strategy)
