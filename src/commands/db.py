import typer

from src.config import (
    BASE_TAG,
    COLLECTION_NAME,
    MONGODB_DB,
    MONGODB_PARENTS_COLLECTION,
    MONGODB_FILES_METADATA_COLLECTION,
)
from src.db.client import get_client
from src.db.collection import setup_collection, get_collection_info
from src.db.mongo import get_mongo_client, get_parents_collection

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
        typer.echo(f"Base tag:        {BASE_TAG}")
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
        typer.echo(f"Base tag:             {BASE_TAG}")
        typer.echo(f"MongoDB:              {MONGODB_DB}.{MONGODB_PARENTS_COLLECTION}")
        typer.echo(f"Liczba parentów:      {count}")
    except Exception as e:
        typer.secho(f"Błąd połączenia z MongoDB: {e}", fg=typer.colors.RED)


@app.command("variants")
def variants():
    """Listuje wszystkie warianty bazy (BASE_TAG) widoczne w Qdrant + MongoDB.

    Wariant istnieje, jeśli ma kolekcję 'documents_<tag>' w Qdrant ALBO bazę 'rag_<tag>' w Mongo.
    Pokazuje liczbę vectorów (Qdrant), parentów i plików (Mongo) per wariant.
    """
    qdrant_collections = {c.name for c in get_client().get_collections().collections}
    mongo_dbs = set(get_mongo_client().list_database_names())

    qdrant_tags = {c.removeprefix("documents_") for c in qdrant_collections if c.startswith("documents_")}
    mongo_tags = {db.removeprefix("rag_") for db in mongo_dbs if db.startswith("rag_")}
    all_tags = sorted(qdrant_tags | mongo_tags)

    if not all_tags:
        typer.echo("Brak wariantów. Ustaw BASE_TAG i uruchom 'db setup' + 'ingest run'.")
        return

    typer.echo(f"{'TAG':<30} {'VECTORS':>10} {'PARENTS':>10} {'FILES':>8}  ACTIVE")
    typer.echo("-" * 75)
    for tag in all_tags:
        col_name = f"documents_{tag}"
        db_name = f"rag_{tag}"

        vectors = "—"
        if col_name in qdrant_collections:
            try:
                info = get_client().get_collection(col_name)
                vectors = str(info.points_count or 0)
            except Exception:
                vectors = "?"

        parents = files = "—"
        if db_name in mongo_dbs:
            try:
                db = get_mongo_client()[db_name]
                parents = str(db[MONGODB_PARENTS_COLLECTION].count_documents({}))
                files = str(db[MONGODB_FILES_METADATA_COLLECTION].count_documents({}))
            except Exception:
                parents = files = "?"

        active = "← active" if tag == BASE_TAG else ""
        typer.echo(f"{tag:<30} {vectors:>10} {parents:>10} {files:>8}  {active}")


@app.command("drop")
def drop(
    tag: str = typer.Argument(..., help="BASE_TAG wariantu do usunięcia"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Pomiń potwierdzenie"),
):
    """Usuwa wariant bazy: kolekcję 'documents_<tag>' w Qdrant i bazę 'rag_<tag>' w Mongo.

    UWAGA: nieodwracalne. Bazy z plikami źródłowymi (./DOKUMENTY) NIE są ruszane.
    """
    if tag == BASE_TAG:
        typer.secho(
            f"Odmowa: '{tag}' to aktywny wariant (BASE_TAG). Przełącz się na inny "
            "wariant przed usunięciem (zmień BASE_TAG w .env i restart rag-server).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    col_name = f"documents_{tag}"
    db_name = f"rag_{tag}"

    qdrant_collections = {c.name for c in get_client().get_collections().collections}
    mongo_dbs = set(get_mongo_client().list_database_names())

    has_qdrant = col_name in qdrant_collections
    has_mongo = db_name in mongo_dbs

    if not has_qdrant and not has_mongo:
        typer.secho(f"Wariant '{tag}' nie istnieje (brak '{col_name}' i '{db_name}').", fg=typer.colors.YELLOW)
        return

    typer.echo(f"Wariant '{tag}' do usunięcia:")
    if has_qdrant:
        typer.echo(f"  • Qdrant collection: {col_name}")
    if has_mongo:
        typer.echo(f"  • MongoDB database:  {db_name}")

    if not yes:
        confirmed = typer.confirm("Usunąć?")
        if not confirmed:
            typer.echo("Anulowano.")
            raise typer.Exit(code=0)

    if has_qdrant:
        get_client().delete_collection(col_name)
        typer.secho(f"  ✓ Usunięto Qdrant collection '{col_name}'", fg=typer.colors.GREEN)
    if has_mongo:
        get_mongo_client().drop_database(db_name)
        typer.secho(f"  ✓ Usunięto MongoDB database '{db_name}'", fg=typer.colors.GREEN)
