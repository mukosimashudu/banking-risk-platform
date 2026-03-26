import pandas as pd
from pathlib import Path
from src.config.db import engine

BASE_DIR = Path(__file__).resolve().parents[2]
FILE_PATH = BASE_DIR / "data/raw/credit/cs-training.csv"

def main():
    print("Reading CSV...")

    df = pd.read_csv(FILE_PATH)

    print("CSV Shape:", df.shape)

    if df.empty:
        raise ValueError("CSV is empty ❌")

    print("Loading into SQL...")

    df.to_sql(
        name="credit_scoring_data",
        con=engine,
        schema="raw",
        if_exists="replace",
        index=False,
        chunksize=1000,
        method="multi"
    )

    print("✅ Data loaded successfully.")

if __name__ == "__main__":
    main()