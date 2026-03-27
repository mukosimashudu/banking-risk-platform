from pathlib import Path
import traceback
from datetime import datetime
from typing import List

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# ==============================
# DATABASE (EDIT THIS)
# ==============================
DB_CONNECTION = "mssql+pyodbc://YOUR_SERVER/YOUR_DB?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes"

try:
    engine = create_engine(DB_CONNECTION)
    DB_ENABLED = True
except Exception as e:
    print("DB ERROR:", e)
    engine = None
    DB_ENABLED = False

# ==============================
# PATHS + MODELS
# ==============================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

fraud_model = joblib.load(MODEL_DIR / "fraud_model_best.pkl")
credit_model = joblib.load(MODEL_DIR / "credit_model_best.pkl")

explainer = shap.Explainer(credit_model)

# ==============================
# APP
# ==============================
app = FastAPI(title="Banking Risk API")

# ==============================
# REQUEST MODEL
# ==============================
class LoanApplication(BaseModel):
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

# ==============================
# FEATURE BUILDERS
# ==============================
def fraud_df(d):
    return pd.DataFrame([{
        "transaction_amt": d.transaction_amt,
        "card1": d.card1,
        "card2": d.card2,
        "card3": d.card3,
        "card5": d.card5,
        "addr1": d.addr1,
        "addr2": d.addr2
    }])

def credit_df(d):
    return pd.DataFrame([{
        "utilization": d.utilization,
        "age": d.age,
        "late_30_59": d.late_30_59,
        "debt_ratio": d.debt_ratio,
        "income": d.income,
        "open_loans": d.open_credit,
        "late_90": d.late_90,
        "real_estate_loans": d.real_estate,
        "late_60_89": d.late_60_89,
        "dependents": d.dependents
    }])

# ==============================
# SHAP
# ==============================
def get_shap(df):
    try:
        shap_values = explainer(df)
        vals = shap_values.values[0]

        df2 = pd.DataFrame({
            "feature": df.columns,
            "impact": vals
        })

        df2["abs"] = df2["impact"].abs()
        df2 = df2.sort_values("abs", ascending=False).head(5)

        return df2[["feature", "impact"]].to_dict(orient="records")
    except:
        return []

# ==============================
# DECISION
# ==============================
def decide(d, fraud, credit):
    disposable = d.income - d.monthly_expenses

    if fraud > 0.7:
        return "REJECT - FRAUD"
    if credit > 0.5:
        return "REJECT - DEFAULT"
    if disposable < 1000:
        return "REJECT - LOW INCOME"
    return "APPROVE"

# ==============================
# SAVE TO SQL
# ==============================
def save(d, fraud, credit, dec):
    if not DB_ENABLED:
        return

    try:
        with engine.begin() as conn:
            conn.execute(text("""
            INSERT INTO prediction_log
            (requested_amount, fraud_score, probability_default, decision, created_at)
            VALUES (:amt, :fraud, :credit, :dec, :dt)
            """), {
                "amt": d.transaction_amt,
                "fraud": fraud,
                "credit": credit,
                "dec": dec,
                "dt": datetime.now()
            })
    except Exception as e:
        print("SQL ERROR:", e)

# ==============================
# ROUTES
# ==============================
@app.get("/")
def home():
    return {"status": "running", "db": DB_ENABLED}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(d: LoanApplication):
    try:
        f = float(fraud_model.predict_proba(fraud_df(d))[0][1])
        c = float(credit_model.predict_proba(credit_df(d))[0][1])

        dec = decide(d, f, c)
        shap_data = get_shap(credit_df(d))

        save(d, f, c, dec)

        return {
            "fraud_probability": f,
            "credit_probability": c,
            "decision": dec,
            "top_features": shap_data
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/kpi/summary")
def kpi():
    if not DB_ENABLED:
        return {}

    row = engine.execute(text("""
        SELECT COUNT(*) total,
        SUM(CASE WHEN decision='APPROVE' THEN 1 ELSE 0 END) approved,
        AVG(fraud_score) avg_fraud,
        AVG(probability_default) avg_credit
        FROM prediction_log
    """)).fetchone()

    return dict(row)

@app.post("/chat/query")
def chat(q: dict):
    if not DB_ENABLED:
        return {"answer": "DB not connected"}

    if "approved" in q["question"].lower():
        row = engine.execute(text("SELECT COUNT(*) total FROM prediction_log WHERE decision='APPROVE'")).fetchone()
        return {"answer": dict(row)}

    return {"answer": "Ask about approvals"}