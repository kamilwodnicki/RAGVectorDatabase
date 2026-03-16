import typer
import uvicorn
from src.commands import db, benchmark, speedtest

app = typer.Typer(help="Profesjonalny System Zarządzania RAG dla pracy magisterskiej")

app.add_typer(db.app, name="db")
app.add_typer(benchmark.app, name="test")
app.add_typer(speedtest.app, name="speedtest")

@app.command("serve")
def serve(
    port: int = typer.Option(8000, help="Port, na którym ma działać API"),
    host: str = typer.Option("0.0.0.0", help="Host serwera")
):
    """Uruchamia serwer FastAPI (obsługujący zapytania na CPU)."""
    typer.echo(f"Uruchamiam serwer API na {host}:{port}...")
    uvicorn.run("src.api:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    app()