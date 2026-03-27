from pathlib import Path
import traceback
from datetime import datetime
from typing import List

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text


# =========================================================
# DATABASE
# =========================================================
DB_ENABLED = True
DB_ERROR = None
engine = None

try:
    from src.config.db import engine
except Exception as e:
    DB_ENABLED = False
    DB_ERROR = str(e)


# =========================================================
# PATHS / MODELS
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

fraud_model = joblib.load(MODEL_DIR / "fraud_model_best.pkl")
credit_model = joblib.load(MODEL_DIR / "credit_model_best.pkl")

# 🔥 SHAP EXPLAINER (LOAD ONCE)
credit_explainer = shap.Explainer(credit_model)


# =========================================================
# FASTAPI
# =========================================================
app = FastAPI(
    title="Banking Risk API",
    version="3.0"
)


# =========================================================
# REQUEST MODEL
# =========================================================
class LoanApplicationRequest(BaseModel):
    transaction_amt: float
    card1: float
    card2: float
    card3: float
    card5: float
    addr1: float
    addr2: float

    utilization: float
    age: float
    late_30_59: float
    debt_ratio: float
    income: float
    open_credit: float
    late_90: float
    real_estate: float
    late_60_89: float
    dependents: float

    monthly_expenses: float = 0
    marital_status: str = "single"


# =========================================================
# FEATURE BUILDERS
# =========================================================
def build_fraud_features(data):
    return pd.DataFrame([{
        "transaction_amt": data.transaction_amt,
        "card1": data.card1,
        "card2": data.card2,
        "card3": data.card3,
        "card5": data.card5,
        "addr1": data.addr1,
        "addr2": data.addr2,
    }])


def build_credit_features(data):
    return pd.DataFrame([{
        "utilization": data.utilization,
        "age": data.age,
        "late_30_59": data.late_30_59,
        "debt_ratio": data.debt_ratio,
        "income": data.income,
        "open_loans": data.open_credit,
        "late_90": data.late_90,
        "real_estate_loans": data.real_estate,
        "late_60_89": data.late_60_89,
        "dependents": data.dependents,
    }])


# =========================================================
# SHAP FUNCTION
# =========================================================
def get_top_shap_features(df):
    try:
        shap_values = credit_explainer(df)

        shap_df = pd.DataFrame({
            "feature": df.columns,
            "impact": shap_values.values[0]
        })

        shap_df["abs"] = shap_df["impact"].abs()
        shap_df = shap_df.sort_values("abs", ascending=False).head(3)

        return shap_df[["feature", "impact"]].to_dict(orient="records")

    except Exception as e:
        return []


# =========================================================
# DECISION LOGIC
# =========================================================
def decision_logic(data, fraud_prob, credit_prob):
    if fraud_prob > 0.7:
        return "REJECT - FRAUD"

    if credit_prob > 0.5:
        return "REJECT - DEFAULT"

    if data.debt_ratio > 0.6:
        return "REJECT - HIGH DEBT"

    return "APPROVE"


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def home():
    return {
        "message": "Banking Risk API is running",
        "database_enabled": DB_ENABLED,
        "database_error": DB_ERROR
    }


@app.post("/predict")
def predict(data: LoanApplicationRequest):
    try:
        fraud_df = build_fraud_features(data)
        credit_df = build_credit_features(data)

        fraud_prob = float(fraud_model.predict_proba(fraud_df)[0][1])
        credit_prob = float(credit_model.predict_proba(credit_df)[0][1])

        decision = decision_logic(data, fraud_prob, credit_prob)

        # 🔥 SHAP HERE
        top_features = get_top_shap_features(credit_df)

        return {
            "fraud_probability": round(fraud_prob, 4),
            "credit_probability": round(credit_prob, 4),
            "decision": decision,
            "top_features": top_features
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))