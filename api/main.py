from pathlib import Path
from datetime import datetime
from typing import List
from urllib.parse import quote_plus
import os
import traceback

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text


# =========================================================
# APP
# =========================================================
app = FastAPI(title="Banking Risk API", version="1.0.0")


# =========================================================
# DATABASE CONFIG
# =========================================================
DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")

engine = None
DB_ENABLED = False


def init_engine():
    global engine, DB_ENABLED

    if not all([DB_SERVER, DB_DATABASE, DB_USERNAME, DB_PASSWORD]):
        print("Database environment variables not fully set. Running without DB.")
        engine = None
        DB_ENABLED = False
        return

    try:
        connection_string = (
            f"DRIVER={{{DB_DRIVER}}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_DATABASE};"
            f"UID={DB_USERNAME};"
            f"PWD={DB_PASSWORD};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )

        params = quote_plus(connection_string)

        engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={params}",
            fast_executemany=True,
            pool_pre_ping=True,
        )

        DB_ENABLED = True
        print("Database engine initialized successfully.")

    except Exception as e:
        print("DB INIT ERROR:", e)
        engine = None
        DB_ENABLED = False


def ensure_prediction_log_table():
    if not DB_ENABLED or engine is None:
        return

    create_sql = """
    IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'ml')
    BEGIN
        EXEC('CREATE SCHEMA ml')
    END;

    IF OBJECT_ID('ml.prediction_log', 'U') IS NULL
    BEGIN
        CREATE TABLE ml.prediction_log (
            id INT IDENTITY(1,1) PRIMARY KEY,
            requested_amount FLOAT NULL,
            fraud_score FLOAT NULL,
            probability_default FLOAT NULL,
            decision NVARCHAR(100) NULL,
            created_at DATETIME2 NOT NULL DEFAULT SYSDATETIME()
        );
    END;
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(create_sql))
        print("prediction_log table ready.")
    except Exception as e:
        print("TABLE INIT ERROR:", e)


@app.on_event("startup")
def startup_event():
    init_engine()
    ensure_prediction_log_table()


# =========================================================
# MODELS
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

fraud_model = joblib.load(MODEL_DIR / "fraud_model_best.pkl")
credit_model = joblib.load(MODEL_DIR / "credit_model_best.pkl")

try:
    credit_explainer = shap.Explainer(credit_model)
except Exception:
    credit_explainer = None


# =========================================================
# REQUEST / RESPONSE SCHEMAS
# =========================================================
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

    monthly_expenses: float = 0.0
    marital_status: str = "single"


class ChatQuery(BaseModel):
    question: str


# =========================================================
# FEATURE BUILDERS
# =========================================================
def fraud_df(d: LoanApplication) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "transaction_amt": d.transaction_amt,
            "card1": d.card1,
            "card2": d.card2,
            "card3": d.card3,
            "card5": d.card5,
            "addr1": d.addr1,
            "addr2": d.addr2,
        }]
    )


def credit_df(d: LoanApplication) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "utilization": d.utilization,
            "age": d.age,
            "late_30_59": d.late_30_59,
            "debt_ratio": d.debt_ratio,
            "income": d.income,
            "open_loans": d.open_credit,
            "late_90": d.late_90,
            "real_estate_loans": d.real_estate,
            "late_60_89": d.late_60_89,
            "dependents": d.dependents,
        }]
    )


# =========================================================
# EXPLAINABILITY
# =========================================================
def get_shap(df: pd.DataFrame) -> list:
    if credit_explainer is None:
        return []

    try:
        shap_values = credit_explainer(df)
        values = shap_values.values[0]

        explain_df = pd.DataFrame(
            {
                "feature": df.columns,
                "input_value": df.iloc[0].values,
                "impact": values,
            }
        )

        explain_df["abs_impact"] = explain_df["impact"].abs()
        explain_df = explain_df.sort_values("abs_impact", ascending=False).head(5)

        return explain_df[["feature", "input_value", "impact"]].to_dict(orient="records")

    except Exception as e:
        print("SHAP ERROR:", e)
        return []


# =========================================================
# DECISION ENGINE
# =========================================================
def decide(application: LoanApplication, fraud_prob: float, credit_prob: float) -> str:
    disposable_income = application.income - application.monthly_expenses

    if fraud_prob > 0.70:
        return "REJECT - FRAUD"

    if credit_prob > 0.50:
        return "REJECT - DEFAULT"

    if disposable_income < 1000:
        return "REJECT - LOW INCOME"

    return "APPROVE"


# =========================================================
# PERSISTENCE
# =========================================================
def save_prediction(application: LoanApplication, fraud_prob: float, credit_prob: float, decision: str) -> None:
    if not DB_ENABLED or engine is None:
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ml.prediction_log
                    (requested_amount, fraud_score, probability_default, decision, created_at)
                    VALUES (:amt, :fraud, :credit, :decision, :created_at)
                    """
                ),
                {
                    "amt": application.transaction_amt,
                    "fraud": fraud_prob,
                    "credit": credit_prob,
                    "decision": decision,
                    "created_at": datetime.now(),
                },
            )
    except Exception as e:
        print("SAVE ERROR:", e)


# =========================================================
# CORE SCORING
# =========================================================
def score_one(application: LoanApplication) -> dict:
    fraud_probability = float(fraud_model.predict_proba(fraud_df(application))[0][1])
    credit_probability = float(credit_model.predict_proba(credit_df(application))[0][1])

    decision = decide(application, fraud_probability, credit_probability)
    top_features = get_shap(credit_df(application))

    save_prediction(application, fraud_probability, credit_probability, decision)

    return {
        "fraud_probability": fraud_probability,
        "credit_probability": credit_probability,
        "decision": decision,
        "top_features": top_features,
    }


# =========================================================
# CHAT / ASSISTANT
# =========================================================
def get_db_summary() -> dict:
    if not DB_ENABLED or engine is None:
        return {
            "total_applications": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "avg_credit_probability": 0.0,
            "avg_fraud_probability": 0.0,
            "first_prediction_at": None,
            "last_prediction_at": None,
        }

    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                    COUNT(*) AS total_applications,
                    SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) AS total_approved,
                    SUM(CASE WHEN decision <> 'APPROVE' THEN 1 ELSE 0 END) AS total_rejected,
                    AVG(probability_default) AS avg_credit_probability,
                    AVG(fraud_score) AS avg_fraud_probability,
                    MIN(created_at) AS first_prediction_at,
                    MAX(created_at) AS last_prediction_at
                FROM ml.prediction_log
                """
            )
        ).fetchone()

    data = dict(row._mapping)

    for key in ["total_applications", "total_approved", "total_rejected"]:
        data[key] = int(data[key] or 0)

    for key in ["avg_credit_probability", "avg_fraud_probability"]:
        data[key] = float(data[key] or 0.0)

    return data


def answer_chat_question(question: str) -> str:
    q = question.strip().lower()
    summary = get_db_summary()

    total = summary["total_applications"]
    approved = summary["total_approved"]
    rejected = summary["total_rejected"]
    avg_credit = summary["avg_credit_probability"]
    avg_fraud = summary["avg_fraud_probability"]

    if total == 0:
        return "There are no scored applications in the database yet."

    if "approved" in q:
        return f"Total approved applications: {approved} out of {total}."

    if "rejected" in q:
        return f"Total rejected applications: {rejected} out of {total}."

    if "total" in q or "applications" in q:
        return f"Total applications scored: {total}."

    if "fraud" in q:
        return f"Average fraud probability is {avg_fraud:.4f}."

    if "credit" in q or "default" in q:
        return f"Average credit probability is {avg_credit:.4f}."

    if "qualify" in q or "approval rate" in q:
        approval_rate = approved / total if total else 0.0
        return f"Approval rate is {approval_rate:.2%}. Approved: {approved}, rejected: {rejected}."

    return (
        f"Summary: total applications = {total}, approved = {approved}, "
        f"rejected = {rejected}, average fraud probability = {avg_fraud:.4f}, "
        f"average credit probability = {avg_credit:.4f}."
    )


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Banking Risk API",
        "database_enabled": DB_ENABLED,
    }


@app.get("/health")
def health():
    db_status = "disabled"

    if DB_ENABLED and engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "database": db_status,
    }


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        return score_one(application)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def batch_predict(applications: List[LoanApplication]):
    try:
        results = [score_one(application) for application in applications]
        return {
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kpi/summary")
def kpi_summary():
    return get_db_summary()


@app.get("/monitoring/summary")
def monitoring_summary():
    summary = get_db_summary()
    return {
        "total_predictions": summary["total_applications"],
        "avg_fraud": summary["avg_fraud_probability"],
        "avg_credit": summary["avg_credit_probability"],
        "first_prediction_at": summary["first_prediction_at"],
        "last_prediction_at": summary["last_prediction_at"],
    }


@app.post("/chat/query")
def chat_query(payload: ChatQuery):
    try:
        return {"answer": answer_chat_question(payload.question)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))