from pathlib import Path
import joblib
import pandas as pd

from src.scoring.explain import explain_credit
from src.scoring.decision_engine import make_decision

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

fraud_model = joblib.load(MODEL_DIR / "fraud_model.pkl")
credit_model = joblib.load(MODEL_DIR / "credit_model.pkl")

def score_application(payload: dict) -> dict:
    fraud_input = pd.DataFrame([{
        "TransactionAmt": payload["requested_amount"],
        "card1": payload.get("card1", 1000),
        "card2": payload.get("card2", 100),
        "card3": payload.get("card3", 150),
        "card5": payload.get("card5", 200),
        "addr1": payload.get("addr1", 300),
        "addr2": payload.get("addr2", 87),
        "amount_log": payload["requested_amount"]
    }])

    credit_input = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": payload["revolving_utilization"],
        "age": payload["age"],
        "NumberOfTime30_59DaysPastDueNotWorse": payload["late_30_59"],
        "DebtRatio": payload["debt_ratio"],
        "MonthlyIncome": payload["monthly_income"],
        "NumberOfOpenCreditLinesAndLoans": payload["open_credit_lines"],
        "NumberOfTimes90DaysLate": payload["late_90"],
        "NumberRealEstateLoansOrLines": payload["real_estate_loans"],
        "NumberOfTime60_89DaysPastDueNotWorse": payload["late_60_89"],
        "NumberOfDependents": payload["dependents"],
        "income_to_debt_proxy": payload["monthly_income"] / (payload["debt_ratio"] + 1)
    }])

    fraud_score = float(fraud_model.predict_proba(fraud_input)[0][1])
    probability_default = float(credit_model.predict_proba(credit_input)[0][1])

    decision = make_decision(probability_default, fraud_score)

    credit_explanation = explain_credit(credit_input)

    return {
        "probability_default": probability_default,
        "fraud_score": fraud_score,
        "decision": decision,
        "explanation": credit_explanation
    }
    