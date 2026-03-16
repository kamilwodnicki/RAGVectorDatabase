# src/core/persistence.py
import json
import os
from .config import CONFIG_PATH, DEFAULT_DB

def save_active_db(db_type: str, db_variant: str):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump({"db_type": db_type, "db_variant": db_variant}, f)

def get_active_db():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_DB
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)