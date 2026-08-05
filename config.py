from pathlib import Path
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "support_vector_regression.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "dataset", "train.csv"
)

KEY = "house_price_secret_key"