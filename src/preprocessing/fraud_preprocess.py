import pandas as pd

def preprocess_fraud(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    selected = ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "isFraud"]
    data = data[selected]

    for col in data.columns:
        if col != "isFraud":
            data[col] = data[col].fillna(data[col].median())

    return data