import pandas as pd

def preprocess_credit(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    rename_map = {
        "NumberOfTime30-59DaysPastDueNotWorse": "NumberOfTime30_59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse": "NumberOfTime60_89DaysPastDueNotWorse"
    }

    data = data.rename(columns=rename_map)

    selected = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30_59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60_89DaysPastDueNotWorse",
        "NumberOfDependents",
        "SeriousDlqin2yrs"
    ]

    data = data[selected]

    for col in data.columns:
        if col != "SeriousDlqin2yrs":
            data[col] = data[col].fillna(data[col].median())

    return data