from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text

from src.config.db import engine
from src.llm.llm_assistant import generate_explanation


app = FastAPI(title="Full Fintech Banking Platform API")


# =========================================================
# HELPERS
# =========================================================
def make_reference(prefix: str) -> str:
    return f"{prefix}-{str(uuid.uuid4())[:8].upper()}"


# =========================================================
# DATABASE INIT
# =========================================================
def init_schema():
    if engine is None:
        print("⚠️ DB not available — skipping schema creation")
        return

    sql = """
    IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name='ml')
    EXEC('CREATE SCHEMA ml');

    IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name='analytics')
    EXEC('CREATE SCHEMA analytics');

    IF OBJECT_ID('ml.prediction_log') IS NULL
    CREATE TABLE ml.prediction_log (
        id INT IDENTITY PRIMARY KEY,
        application_reference NVARCHAR(100),
        customer_name NVARCHAR(255),
        decision_type NVARCHAR(50),
        product_type NVARCHAR(100),
        requested_amount FLOAT,
        approved_amount FLOAT,
        approved_limit FLOAT,
        monthly_payment FLOAT,
        net_monthly_income FLOAT,
        existing_debt_payments FLOAT,
        credit_score INT,
        debt_to_income_ratio FLOAT,
        fraud_score FLOAT,
        risk_probability FLOAT,
        final_decision NVARCHAR(50),
        llm_explanation NVARCHAR(MAX),
        created_at DATETIME DEFAULT GETDATE()
    );
    """

    with engine.begin() as conn:
        conn.execute(text(sql))


init_schema()


# =========================================================
# MODELS
# =========================================================
class LoanRequest(BaseModel):
    customer_name: str
    requested_amount: float
    net_monthly_income: float
    existing_debt_payments: float
    credit_score: int
    fraud_score: float = 0.05


class CreditRequest(BaseModel):
    customer_name: str
    net_monthly_income: float
    existing_debt_payments: float
    credit_score: int
    fraud_score: float = 0.05


# =========================================================
# CORE LOGIC
# =========================================================
def assess_loan(req: LoanRequest):
    income = req.net_monthly_income
    debt = req.existing_debt_payments
    dti = debt / income if income > 0 else 1

    risk = min(0.95, max(0.02, (700 - req.credit_score) / 500 + dti + req.fraud_score))
    decision = "APPROVED" if risk < 0.4 else "DECLINED"

    try:
        explanation = generate_explanation({
            "credit_score": req.credit_score,
            "income": income,
            "debt": debt,
            "decision": decision,
            "risk": risk
        })
    except:
        explanation = "Fallback explanation: decision based on risk and affordability."

    return {
        "application_reference": make_reference("LOAN"),
        "customer_name": req.customer_name,
        "decision_type": "loan",
        "product_type": "loan",
        "requested_amount": req.requested_amount,
        "approved_amount": req.requested_amount if decision == "APPROVED" else 0,
        "approved_limit": 0,
        "monthly_payment": req.requested_amount / 12,
        "net_monthly_income": income,
        "existing_debt_payments": debt,
        "credit_score": req.credit_score,
        "debt_to_income_ratio": dti,
        "fraud_score": req.fraud_score,
        "risk_probability": risk,
        "final_decision": decision,
        "llm_explanation": explanation
    }


def assess_credit(req: CreditRequest):
    income = req.net_monthly_income
    debt = req.existing_debt_payments
    dti = debt / income if income > 0 else 1

    risk = min(0.95, max(0.02, (700 - req.credit_score) / 500 + dti + req.fraud_score))
    decision = "APPROVED" if risk < 0.4 else "DECLINED"

    try:
        explanation = generate_explanation({
            "credit_score": req.credit_score,
            "income": income,
            "debt": debt,
            "decision": decision,
            "risk": risk
        })
    except:
        explanation = "Fallback explanation: decision based on risk and affordability."

    return {
        "application_reference": make_reference("CREDIT"),
        "customer_name": req.customer_name,
        "decision_type": "credit",
        "product_type": "credit",
        "requested_amount": 0,
        "approved_amount": 0,
        "approved_limit": income * 4 if decision == "APPROVED" else 0,
        "monthly_payment": 0,
        "net_monthly_income": income,
        "existing_debt_payments": debt,
        "credit_score": req.credit_score,
        "debt_to_income_ratio": dti,
        "fraud_score": req.fraud_score,
        "risk_probability": risk,
        "final_decision": decision,
        "llm_explanation": explanation
    }


# =========================================================
# SAVE
# =========================================================
def save_application(data: Dict):
    if engine is None:
        print("⚠️ DB not available")
        return

    sql = """
    INSERT INTO ml.prediction_log (
        application_reference, customer_name, decision_type,
        requested_amount, approved_amount, approved_limit,
        risk_probability, fraud_score, final_decision, llm_explanation
    )
    VALUES (
        :application_reference, :customer_name, :decision_type,
        :requested_amount, :approved_amount, :approved_limit,
        :risk_probability, :fraud_score, :final_decision, :llm_explanation
    )
    """

    data.setdefault("approved_limit", 0)
    data.setdefault("approved_amount", 0)
    data.setdefault("requested_amount", 0)

    try:
        with engine.begin() as conn:
            conn.execute(text(sql), data)
    except Exception as e:
        print(f"DB ERROR: {e}")


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def home():
    return {"message": "API running"}


@app.get("/health")
def health():
    if engine is None:
        return {"status": "error", "db": "not configured"}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "db": str(e)}


@app.post("/api/loan/assess")
def loan(req: LoanRequest):
    try:
        result = assess_loan(req)
        save_application(result)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/credit/assess")
def credit(req: CreditRequest):
    try:
        result = assess_credit(req)
        save_application(result)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/summary")
def summary():
    if engine is None:
        return {}

    sql = """
    SELECT COUNT(*) as total,
           SUM(CASE WHEN final_decision='APPROVED' THEN 1 ELSE 0 END) as approved
    FROM ml.prediction_log
    """

    with engine.connect() as conn:
        row = conn.execute(text(sql)).fetchone()

    return {"total": row[0], "approved": row[1]}


@app.get("/api/portfolio/recent-loans")
def recent_loans():
    if engine is None:
        return []

    sql = "SELECT TOP 10 * FROM ml.prediction_log WHERE decision_type='loan' ORDER BY id DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()

    return list(rows)


@app.get("/api/portfolio/recent-credit")
def recent_credit():
    if engine is None:
        return []

    sql = "SELECT TOP 10 * FROM ml.prediction_log WHERE decision_type='credit' ORDER BY id DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()

    return list(rows)