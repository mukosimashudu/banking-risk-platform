import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

fraud_model = joblib.load(BASE_DIR / "models/fraud_model_best.pkl")
credit_model = joblib.load(BASE_DIR / "models/credit_model_best.pkl")


def predict_fraud(features):
    prob = fraud_model.predict_proba([features])[0][1]
    return prob


def predict_credit(features):
    prob = credit_model.predict_proba([features])[0][1]
    return prob


def make_decision(fraud_features, credit_features):
    fraud_prob = predict_fraud(fraud_features)
    credit_prob = predict_credit(credit_features)

    if fraud_prob > 0.7:
        decision = "REJECT - FRAUD RISK"
    elif credit_prob > 0.5:
        decision = "REJECT - CREDIT RISK"
    else:
        decision = "APPROVE"

    return {
        "fraud_probability": round(fraud_prob, 4),
        "credit_probability": round(credit_prob, 4),
        "decision": decision
    }