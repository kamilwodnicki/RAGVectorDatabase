import typer
import uvicorn
from src.commands import db, ingest
from src.config import format_effective_config
from src.logging_config import setup_logging

setup_logging()

app = typer.Typer(help="System Zarządzania RAG")

app.add_typer(db.app, name="db")
app.add_typer(ingest.app, name="ingest")


@app.command("serve")
def serve(
    port: int = typer.Option(8000, help="Port serwera API"),
    host: str = typer.Option("0.0.0.0", help="Host serwera"),
):
    typer.echo(format_effective_config())
    typer.echo(f"Uruchamiam serwer API na {host}:{port}...")
    uvicorn.run("src.server.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
