from pathlib import Path
import pandas as pd
from src.config.db import engine

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "credit"
CREDIT_FILE = RAW_DIR / "cs-training.csv"

def main():
    print("Reading credit scoring data...")
    df = pd.read_csv(CREDIT_FILE)

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})

    selected_columns = [
        "id",
        "SeriousDlqin2yrs",
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ]

    rename_map = {
        "NumberOfTime30-59DaysPastDueNotWorse": "NumberOfTime30_59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse": "NumberOfTime60_89DaysPastDueNotWorse"
    }

    df = df[selected_columns].rename(columns=rename_map)

    print(f"Credit shape: {df.shape}")
    print("Writing credit scoring raw table to SQL Server...")
    df.to_sql("credit_scoring_data", con=engine, schema="raw", if_exists="replace", index=False)

    print("Credit raw data loaded successfully.")

if __name__ == "__main__":
    main()