import pandas as pd

def build_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["income_to_debt_proxy"] = data["MonthlyIncome"] / (data["DebtRatio"] + 1)
    return data