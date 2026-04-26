import typer

from src.config import (
    COLLECTION_NAME,
    EXTRACTION_STRATEGY,
    MONGODB_DB,
    MONGODB_FILES_METADATA_COLLECTION,
    MONGODB_PARENTS_COLLECTION,
    PDF_SOURCE_DIR,
)
from src.db.metadata_store import MetadataStoreError
from src.ingest.pipeline import run_rebuild, run_single_file, run_sync

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


@app.command("file")
def file_cmd(
    path: str = typer.Argument(..., help="Ścieżka do pliku do (re)przetworzenia"),
    strategy: str = typer.Option(
        EXTRACTION_STRATEGY,
        "--strategy",
        help="Strategia ekstrakcji PDF (override): fast | hi_res. Domyślnie z .env.",
    ),
):
    """Wymusza reprocess jednego pliku — niezależnie od hash w metadata store.

    Przykłady:
        python manage.py ingest file ./DOKUMENTY/raport.pdf
        python manage.py ingest file ./DOKUMENTY/skan.pdf --strategy hi_res
    """
    try:
        result = run_single_file(path, strategy=strategy)
    except FileNotFoundError as e:
        typer.secho(f"BŁĄD: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.secho(f"BŁĄD: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except RuntimeError as e:
        typer.secho(f"BŁĄD: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except MetadataStoreError as e:
        typer.secho(f"BŁĄD metadata-store: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"BŁĄD ingest: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    label = "REPLACED" if result.replaced_existing else "ADDED"
    typer.secho(
        f"OK [{label}] {path} ({len(result.parent_ids)}p / {len(result.child_ids)}c, strategy={strategy})",
        fg=typer.colors.GREEN,
    )
