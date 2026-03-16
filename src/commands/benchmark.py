import typer
import json
import requests
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from src.core.config import SPLITTER_TYPES, LENGTH_VARIANTS, MODEL_NAME

app = typer.Typer()

EVAL_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
SIMILARITY_THRESHOLD = 0.45 
TEST_FILE = "test_dataset.json"

@app.command("run")
def run_benchmark(
    db_type: str = typer.Option("T4_Smart", "--type", "-t"),
    db_len: str = typer.Option("L3_800", "--variant", "-v"),
    api_url: str = typer.Option("http://127.0.0.1:8000/query/", "--url")
):
    """Uruchamia test semantyczny dla konkretnej bazy."""
    typer.echo(f"--- URUCHAMIAM BENCHMARK: {db_type}/{db_len} ---")
    
    evaluator = SentenceTransformer(EVAL_MODEL_NAME)
    
    try:
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
    except FileNotFoundError:
        typer.secho(f"Błąd: Brak pliku {TEST_FILE}", fg=typer.colors.RED)
        return

    hits = 0
    scores = []

    for case in tqdm(test_set, desc="Testowanie"):
        truth_text = case.get("expected_snippet", "")
        truth_embedding = evaluator.encode(truth_text, convert_to_tensor=True)

        payload = {
            "query": case["question"],
            "db_type": db_type,
            "db_variant": db_len,
            "k": 5
        }
        
        try:
            resp = requests.post(api_url, json=payload)
            if resp.status_code != 200: continue
            
            data = resp.json()
            retrieved_texts = [doc['content'] for doc in data.get("results", [])]
            
            if not retrieved_texts:
                scores.append(0)
                continue

            ret_embeddings = evaluator.encode(retrieved_texts, convert_to_tensor=True)
            cosine_scores = util.cos_sim(truth_embedding, ret_embeddings)[0]
            best_score = float(torch.max(cosine_scores))
            
            scores.append(best_score)
            if best_score >= SIMILARITY_THRESHOLD:
                hits += 1
        except Exception as e:
            typer.echo(f"Error: {e}")

    avg_score = sum(scores) / len(scores) if scores else 0
    hit_rate = (hits / len(test_set)) * 100
    typer.secho(f"\nWynik dla {db_type}/{db_len}:", fg=typer.colors.GREEN)
    typer.echo(f" - Hit Rate: {hit_rate:.2f}%")
    typer.echo(f" - Avg Semantic Score: {avg_score:.3f}")