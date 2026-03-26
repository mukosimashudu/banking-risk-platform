import pandas as pd

def build_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["amount_log"] = data["TransactionAmt"].apply(lambda x: 0 if x <= 0 else x)
    return data