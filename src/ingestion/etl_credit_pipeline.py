from sqlalchemy import text
from src.config.db import engine


def create_clean_credit():
    print("Creating clean credit dataset...")

    query = """
    DROP TABLE IF EXISTS staging.clean_credit;

    SELECT
        TRY_CAST(SeriousDlqin2yrs AS INT) AS target,
        TRY_CAST(RevolvingUtilizationOfUnsecuredLines AS FLOAT) AS utilization,
        TRY_CAST(age AS FLOAT) AS age,
        TRY_CAST(NumberOfTime30_59DaysPastDueNotWorse AS FLOAT) AS late_30_59,
        TRY_CAST(DebtRatio AS FLOAT) AS debt_ratio,
        TRY_CAST(MonthlyIncome AS FLOAT) AS income,
        TRY_CAST(NumberOfOpenCreditLinesAndLoans AS FLOAT) AS open_loans,
        TRY_CAST(NumberOfTimes90DaysLate AS FLOAT) AS late_90,
        TRY_CAST(NumberRealEstateLoansOrLines AS FLOAT) AS real_estate_loans,
        TRY_CAST(NumberOfTime60_89DaysPastDueNotWorse AS FLOAT) AS late_60_89,
        TRY_CAST(NumberOfDependents AS FLOAT) AS dependents
    INTO staging.clean_credit
    FROM raw.credit_scoring_data;
    """

    with engine.begin() as conn:
        conn.execute(text(query))

    print("Credit dataset cleaned.")


if __name__ == "__main__":
    create_clean_credit()