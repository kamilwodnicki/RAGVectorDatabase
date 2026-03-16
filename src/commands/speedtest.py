import time
import typer
import torch
from src.core.engine import E5HuggingFaceEmbeddings

app = typer.Typer()

_BASE_WORDS = (
    "przedsiębiorstwo zarządzanie ekonomia rynek kapitał inwestycje technologia "
    "innowacje strategia konkurencja pracownicy organizacja finanse produkcja "
    "usługi klient wartość zysk koszt budżet analiza raport wynik działalność "
    "sektor branża gospodarka rozwój wzrost efektywność jakość proces system "
)

TEXT_VARIANTS = {
    "bardzo_krotki  (~50 slow)":   50,
    "krotki        (~200 slow)":  200,
    "sredni        (~500 slow)":  500,
    "dlugi        (~1000 slow)": 1000,
}

BATCH_VARIANTS = [1, 10, 50, 100, 500]


def _make_texts(word_count: int, n: int) -> list[str]:
    words = (_BASE_WORDS.split() * ((word_count // len(_BASE_WORDS.split())) + 2))[:word_count]
    text = " ".join(words)
    return [text] * n


@app.command("run")
def run_speedtest(
    num_texts: int = typer.Option(200, "--texts", "-n", help="Liczba tekstów w teście długości"),
    warmup: int = typer.Option(5, "--warmup", "-w", help="Liczba tekstów rozgrzewkowych"),
):
    """Porównuje prędkość embeddingów na CPU vs GPU dla różnych długości i liczby tekstów."""

    devices: list[str] = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        typer.secho("Wykryto GPU — test obejmie CPU i GPU.", fg=typer.colors.GREEN)
    else:
        typer.secho("Brak GPU (CUDA) — test tylko na CPU.", fg=typer.colors.YELLOW)

    results: dict[str, dict[str, tuple[float, float]]] = {}

    for device in devices:
        typer.secho(f"\n{'='*60}", fg=typer.colors.CYAN)
        typer.secho(f"  Ładowanie modelu na: {device.upper()}", fg=typer.colors.CYAN)
        typer.secho(f"{'='*60}", fg=typer.colors.CYAN)
        model = E5HuggingFaceEmbeddings(device=device)
        results[device] = {}

        typer.secho("\n[TEST 1] Wpływ długości tekstu (n=%d tekstów)" % num_texts,
                    fg=typer.colors.BRIGHT_WHITE)
        typer.echo(f"  {'Wariant':<30} {'Czas (s)':>10} {'Tekstów/s':>12}")
        typer.echo(f"  {'-'*54}")

        warmup_texts = _make_texts(50, warmup)
        model.embed_documents(warmup_texts) 

        for label, wc in TEXT_VARIANTS.items():
            texts = _make_texts(wc, num_texts)
            t0 = time.perf_counter()
            model.embed_documents(texts)
            elapsed = time.perf_counter() - t0
            tps = num_texts / elapsed
            results[device][label] = (elapsed, tps)
            typer.echo(f"  {label:<30} {elapsed:>10.2f} {tps:>11.1f}")

        typer.secho("\n[TEST 2] Wpływ liczby tekstów (długość ~200 słów)",
                    fg=typer.colors.BRIGHT_WHITE)
        typer.echo(f"  {'Liczba tekstów':<20} {'Czas (s)':>10} {'Tekstów/s':>12} {'ms/tekst':>10}")
        typer.echo(f"  {'-'*54}")

        for n in BATCH_VARIANTS:
            texts = _make_texts(200, n)
            t0 = time.perf_counter()
            model.embed_documents(texts)
            elapsed = time.perf_counter() - t0
            tps = n / elapsed
            ms_per = (elapsed / n) * 1000
            key = f"batch_{n}"
            results[device][key] = (elapsed, tps)
            typer.echo(f"  {n:<20} {elapsed:>10.3f} {tps:>11.1f} {ms_per:>9.2f}")

    if "cuda" in results:
        typer.secho(f"\n{'='*60}", fg=typer.colors.GREEN)
        typer.secho("  PODSUMOWANIE: przyspieszenie GPU względem CPU", fg=typer.colors.GREEN)
        typer.secho(f"{'='*60}", fg=typer.colors.GREEN)
        typer.echo(f"  {'Wariant':<30} {'CPU (s)':>9} {'GPU (s)':>9} {'Speedup':>9}")
        typer.echo(f"  {'-'*60}")
        for key in results["cpu"]:
            cpu_t = results["cpu"][key][0]
            gpu_t = results["cuda"][key][0]
            speedup = cpu_t / gpu_t
            typer.echo(f"  {key:<30} {cpu_t:>9.3f} {gpu_t:>9.3f} {speedup:>8.1f}x")
