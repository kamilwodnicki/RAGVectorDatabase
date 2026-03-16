import os
import torch

BASE_DB_DIR = "./vectorDatabase_MULTI"
CONFIG_PATH = "./config/active_db.json" 
PDF_SOURCE_DIR = "./DOKUMENTY"
MODEL_NAME = 'intfloat/multilingual-e5-base'

API_DEVICE = "cpu"
INGEST_DEVICE = "cuda" if torch.cuda.is_available() else None

SPLITTER_TYPES = ["T1_Sztywny", "T2_Zdania", "T3_ZdaniaContext", "T4_Smart"]
LENGTH_VARIANTS = {
    "L1_200": 200, "L2_500": 500, "L3_800": 800, "L4_1200": 1200, "L5_1600": 1600
}

DEFAULT_DB = {
    "db_type": "T4_Smart",
    "db_variant": "L3_800"
}