import typer
from src.db.collection import setup_collection, get_collection_info
from src.db.mongo import get_parents_collection
from src.config import COLLECTION_NAME, MONGODB_DB, MONGODB_PARENTS_COLLECTION

app = typer.Typer()


@app.command("setup")
def setup(
    recreate: bool = typer.Option(False, "--recreate", help="Usuń i utwórz kolekcję od nowa"),
):
    setup_collection(recreate=recreate)
    typer.secho(f"Kolekcja '{COLLECTION_NAME}' gotowa.", fg=typer.colors.GREEN)


@app.command("status")
def status():
    try:
        info = get_collection_info()
        points = info.points_count if info.points_count is not None else 0
        typer.echo(f"Kolekcja:        {COLLECTION_NAME}")
        typer.echo(f"Liczba chunków:  {points}")
        typer.echo(f"Status:          {info.status}")
    except Exception as e:
        typer.secho(f"Błąd połączenia z Qdrant: {e}", fg=typer.colors.RED)


@app.command("mongo-status")
def mongo_status():
    try:
        col = get_parents_collection()
        count = col.count_documents({})
        typer.echo(f"MongoDB:              {MONGODB_DB}.{MONGODB_PARENTS_COLLECTION}")
        typer.echo(f"Liczba parentów:      {count}")
    except Exception as e:
        typer.secho(f"Błąd połączenia z MongoDB: {e}", fg=typer.colors.RED)
