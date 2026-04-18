import typer
from src.ingest.pipeline import run_ingest
from src.config import PDF_SOURCE_DIR, EXTRACTION_STRATEGY

app = typer.Typer()


@app.command("run")
def run(
    source_dir: str = typer.Option(PDF_SOURCE_DIR, "--source", "-s", help="Folder z dokumentami"),
    strategy: str = typer.Option(EXTRACTION_STRATEGY, "--strategy", help="Strategia ekstrakcji PDF: fast | hi_res"),
):
    run_ingest(source_dir, strategy=strategy)
