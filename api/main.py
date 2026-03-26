from pathlib import Path
import traceback
from datetime import datetime
from typing import List

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text


# =========================================================
# OPTIONAL DATABASE IMPORT
# =========================================================
DB_ENABLED = True
DB_ERROR = None
engine = None

try:
    from src.config.db import engine  # type: ignore
except Exception as e:
    DB_ENABLED = False
    DB_ERROR = str(e)
    print(f"[WARNING] Database disabled at startup: {DB_ERROR}")


# =========================================================
# PATHS / MODELS
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

FRAUD_MODEL_PATH = MODEL_DIR / "fraud_model_best.pkl"
CREDIT_MODEL_PATH = MODEL_DIR / "credit_model_best.pkl"

if not FRAUD_MODEL_PATH.exists():
    raise FileNotFoundError(f"Fraud model not found: {FRAUD_MODEL_PATH}")

if not CREDIT_MODEL_PATH.exists():
    raise FileNotFoundError(f"Credit model not found: {CREDIT_MODEL_PATH}")

fraud_model = joblib.load(FRAUD_MODEL_PATH)
credit_model = joblib.load(CREDIT_MODEL_PATH)


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Banking Risk Decision API",
    version="2.1.0",
    description="Fraud + Credit Risk scoring API with optional SQL logging, KPI endpoints, monitoring, and batch scoring"
)


# =========================================================
# PYDANTIC MODELS
# =========================================================
class LoanApplicationRequest(BaseModel):
    # Fraud features
    transaction_amt: float = Field(..., ge=0)
    card1: float = Field(..., ge=0)
    card2: float = Field(..., ge=0)
    card3: float = Field(..., ge=0)
    card5: float = Field(..., ge=0)
    addr1: float = Field(..., ge=0)
    addr2: float = Field(..., ge=0)

    # Credit features
    utilization: float = Field(..., ge=0)
    age: float = Field(..., ge=18)
    late_30_59: float = Field(..., ge=0)
    debt_ratio: float = Field(..., ge=0)
    income: float = Field(..., ge=0)
    open_credit: float = Field(..., ge=0)
    late_90: float = Field(..., ge=0)
    real_estate: float = Field(..., ge=0)
    late_60_89: float = Field(..., ge=0)
    dependents: float = Field(..., ge=0)

    # Optional business fields
    monthly_expenses: float = Field(0, ge=0)
    marital_status: str = Field("single")


class ChatRequest(BaseModel):
    question: str


# =========================================================
# FEATURE BUILDERS
# =========================================================
def build_fraud_features(data: LoanApplicationRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "transaction_amt": data.transaction_amt,
        "card1": data.card1,
        "card2": data.card2,
        "card3": data.card3,
        "card5": data.card5,
        "addr1": data.addr1,
        "addr2": data.addr2,
    }])


def build_credit_features(data: LoanApplicationRequest) -> pd.DataFrame:
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
# SCORING
# =========================================================
def score_fraud(data: LoanApplicationRequest) -> float:
    df = build_fraud_features(data)
    return float(fraud_model.predict_proba(df)[0][1])


def score_credit(data: LoanApplicationRequest) -> float:
    df = build_credit_features(data)
    return float(credit_model.predict_proba(df)[0][1])


def enhanced_credit_logic(
    data: LoanApplicationRequest,
    fraud_probability: float,
    credit_probability: float
) -> str:
    disposable_income = data.income - data.monthly_expenses

    if fraud_probability > 0.70:
        return "REJECT - FRAUD RISK"

    if data.income < 2000:
        return "REJECT - LOW INCOME"

    if disposable_income < 1000:
        return "REJECT - LOW DISPOSABLE INCOME"

    if data.debt_ratio > 0.60:
        return "REJECT - HIGH DEBT RATIO"

    if data.age < 21:
        return "REJECT - AGE RISK"

    if credit_probability > 0.50:
        return "REJECT - DEFAULT RISK"

    if data.marital_status.lower() not in {"single", "married", "divorced", "widowed"}:
        return "REVIEW - INVALID MARITAL STATUS"

    return "APPROVE"


# =========================================================
# DB HELPERS
# =========================================================
def db_available() -> bool:
    return DB_ENABLED and engine is not None


def save_prediction_to_sql(
    data: LoanApplicationRequest,
    fraud_probability: float,
    credit_probability: float,
    decision: str
) -> None:
    if not db_available():
        print("[INFO] Database not available. Skipping SQL logging.")
        return

    insert_sql = text("""
        INSERT INTO ml.prediction_log (
            customer_id,
            requested_amount,
            probability_default,
            fraud_score,
            decision,
            model_name,
            model_version,
            created_at
        )
        VALUES (
            NULL,
            :requested_amount,
            :probability_default,
            :fraud_score,
            :decision,
            :model_name,
            :model_version,
            :created_at
        )
    """)

    try:
        with engine.begin() as conn:
            conn.execute(
                insert_sql,
                {
                    "requested_amount": data.transaction_amt,
                    "probability_default": credit_probability,
                    "fraud_score": fraud_probability,
                    "decision": decision,
                    "model_name": "fraud_best + credit_best",
                    "model_version": "v2.1",
                    "created_at": datetime.now(),
                }
            )
    except Exception as e:
        print(f"[WARNING] Failed to save prediction to SQL: {e}")


def run_chat_query(question: str):
    if not db_available():
        return {
            "answer": "Database is currently unavailable, so chat metrics are temporarily disabled."
        }

    q = question.lower().strip()

    if "how many approved" in q:
        sql = "SELECT COUNT(*) AS total FROM ml.prediction_log WHERE decision = 'APPROVE'"
    elif "how many rejected" in q:
        sql = "SELECT COUNT(*) AS total FROM ml.prediction_log WHERE decision LIKE 'REJECT%'"
    elif "how many qualify" in q:
        sql = "SELECT COUNT(*) AS total FROM ml.prediction_log WHERE decision = 'APPROVE'"
    elif "total applications" in q or "how many applications" in q:
        sql = "SELECT COUNT(*) AS total FROM ml.prediction_log"
    elif "average fraud" in q:
        sql = "SELECT AVG(fraud_score) AS avg_fraud_score FROM ml.prediction_log"
    elif "average credit" in q or "average default" in q:
        sql = "SELECT AVG(probability_default) AS avg_probability_default FROM ml.prediction_log"
    else:
        return {
            "answer": (
                "I understand questions about approved, rejected, qualified, total applications, "
                "average fraud score, and average default probability."
            )
        }

    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().first()

    return {"answer": dict(row) if row else {}}


# =========================================================
# RESPONSE HELPER
# =========================================================
def build_prediction_response(
    application: LoanApplicationRequest,
    fraud_probability: float,
    credit_probability: float,
    decision: str
) -> dict:
    return {
        "fraud_probability": round(fraud_probability, 4),
        "credit_probability": round(credit_probability, 4),
        "decision": decision,
        "salary": application.income,
        "monthly_expenses": application.monthly_expenses,
        "age": application.age,
        "marital_status": application.marital_status
    }


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def home():
    return {
        "message": "Banking Risk API is running",
        "database_enabled": db_available(),
        "database_error": DB_ERROR if not db_available() else None
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "database_enabled": db_available(),
        "database_error": DB_ERROR if not db_available() else None
    }


@app.get("/kpi/summary")
def kpi_summary():
    if not db_available():
        return {
            "total_applications": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "avg_credit_probability": 0,
            "avg_fraud_probability": 0,
            "note": "Database unavailable. KPI summary is disabled."
        }

    sql = text("""
        SELECT
            COUNT(*) AS total_applications,
            SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) AS total_approved,
            SUM(CASE WHEN decision LIKE 'REJECT%' THEN 1 ELSE 0 END) AS total_rejected,
            AVG(probability_default) AS avg_credit_probability,
            AVG(fraud_score) AS avg_fraud_probability
        FROM ml.prediction_log
    """)

    with engine.connect() as conn:
        row = conn.execute(sql).mappings().first()

    return dict(row) if row else {}


@app.get("/monitoring/summary")
def monitoring_summary():
    if not db_available():
        return {
            "total_predictions": 0,
            "avg_fraud": 0,
            "avg_credit": 0,
            "first_prediction_at": None,
            "last_prediction_at": None,
            "note": "Database unavailable. Monitoring summary is disabled."
        }

    sql = text("""
        SELECT
            COUNT(*) AS total_predictions,
            AVG(fraud_score) AS avg_fraud,
            AVG(probability_default) AS avg_credit,
            MIN(created_at) AS first_prediction_at,
            MAX(created_at) AS last_prediction_at
        FROM ml.prediction_log
    """)

    with engine.connect() as conn:
        row = conn.execute(sql).mappings().first()

    return dict(row) if row else {}


@app.post("/chat/query")
def chat_query(request: ChatRequest):
    try:
        return run_chat_query(request.question)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat query failed: {str(e)}")


@app.post("/predict")
def predict(application: LoanApplicationRequest):
    try:
        fraud_probability = score_fraud(application)
        credit_probability = score_credit(application)
        decision = enhanced_credit_logic(application, fraud_probability, credit_probability)

        save_prediction_to_sql(
            data=application,
            fraud_probability=fraud_probability,
            credit_probability=credit_probability,
            decision=decision
        )

        return build_prediction_response(
            application=application,
            fraud_probability=fraud_probability,
            credit_probability=credit_probability,
            decision=decision
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
def predict_batch(applications: List[LoanApplicationRequest]):
    try:
        results = []

        for application in applications:
            fraud_probability = score_fraud(application)
            credit_probability = score_credit(application)
            decision = enhanced_credit_logic(application, fraud_probability, credit_probability)

            save_prediction_to_sql(
                data=application,
                fraud_probability=fraud_probability,
                credit_probability=credit_probability,
                decision=decision
            )

            results.append(
                build_prediction_response(
                    application=application,
                    fraud_probability=fraud_probability,
                    credit_probability=credit_probability,
                    decision=decision
                )
            )

        return {"results": results, "count": len(results)}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")