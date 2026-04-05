from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.config.db import engine
from src.llm.llm_assistant import generate_explanation


app = FastAPI(title="Full Fintech Banking Platform API")


# =========================================================
# HELPERS
# =========================================================
def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def make_reference(prefix: str) -> str:
    return f"{prefix}-{str(uuid.uuid4())[:8].upper()}"


def fraud_alert_from_score(score: float) -> str:
    if score >= 0.80:
        return "Critical"
    if score >= 0.50:
        return "High"
    if score >= 0.20:
        return "Medium"
    return "Low"


def ifrs9_stage(days_past_due: int, sicr_flag: bool, default_flag: bool) -> str:
    if default_flag or days_past_due >= 90:
        return "Stage 3"
    if sicr_flag or days_past_due >= 30:
        return "Stage 2"
    return "Stage 1"


def build_shap_features(
    *,
    credit_score: int,
    debt_to_income_ratio: float,
    income: float,
    fraud_score: float,
    amount: float,
) -> List[Dict[str, float]]:
    features = [
        {"feature": "credit_score", "shap_value": round((650 - credit_score) / 120, 3)},
        {"feature": "debt_to_income_ratio", "shap_value": round(debt_to_income_ratio, 3)},
        {"feature": "net_monthly_income", "shap_value": round(-(income / 100000), 3)},
        {"feature": "fraud_score", "shap_value": round(fraud_score, 3)},
        {"feature": "requested_amount", "shap_value": round(amount / 500000, 3)},
    ]
    for item in features:
        item["abs_impact"] = abs(item["shap_value"])
    features.sort(key=lambda x: x["abs_impact"], reverse=True)
    return features[:5]


def safe_llm_explanation(
    *,
    credit_score: int,
    income: float,
    debt: float,
    decision: str,
    risk: float,
    dti: float,
    fraud_score: float,
    stage: str,
    decision_type: str,
    shap_features: List[Dict[str, float]],
) -> str:
    fallback = (
        f"{decision_type.title()} decision: {decision}. "
        f"Credit score is {credit_score}, debt-to-income ratio is {dti:.2f}, "
        f"risk probability is {risk:.2%}, fraud score is {fraud_score:.2f}, "
        f"and IFRS 9 stage is {stage}. "
        f"Main drivers include {', '.join([x['feature'] for x in shap_features[:3]])}."
    )

    try:
        explanation = generate_explanation(
            {
                "credit_score": credit_score,
                "income": income,
                "debt": debt,
                "decision": decision,
                "risk": risk,
            }
        )
        return explanation or fallback
    except Exception:
        return fallback


# =========================================================
# DATABASE INITIALISATION
# =========================================================
def init_schema() -> None:
    schema_sql = """
    IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'ml')
        EXEC('CREATE SCHEMA ml');

    IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'analytics')
        EXEC('CREATE SCHEMA analytics');
    """

    prediction_log_sql = """
    IF OBJECT_ID('ml.prediction_log', 'U') IS NULL
    CREATE TABLE ml.prediction_log (
        prediction_id INT IDENTITY(1,1) PRIMARY KEY,
        application_reference NVARCHAR(100) NOT NULL,
        customer_name NVARCHAR(255) NULL,
        decision_type NVARCHAR(50) NOT NULL,
        product_type NVARCHAR(100) NULL,
        requested_amount FLOAT NULL,
        approved_amount FLOAT NULL,
        approved_limit FLOAT NULL,
        monthly_payment FLOAT NULL,
        net_monthly_income FLOAT NULL,
        monthly_expenses FLOAT NULL,
        existing_debt_payments FLOAT NULL,
        credit_score INT NULL,
        debt_to_income_ratio FLOAT NULL,
        fraud_score FLOAT NULL,
        fraud_flag BIT NULL,
        risk_probability FLOAT NULL,
        probability_default FLOAT NULL,
        ifrs9_stage NVARCHAR(50) NULL,
        ecl_lifetime FLOAT NULL,
        final_decision NVARCHAR(50) NULL,
        llm_explanation NVARCHAR(MAX) NULL,
        created_at DATETIME NOT NULL DEFAULT GETDATE()
    );
    """

    fact_sql = """
    IF OBJECT_ID('analytics.fact_applications', 'U') IS NULL
    CREATE TABLE analytics.fact_applications (
        fact_id INT IDENTITY(1,1) PRIMARY KEY,
        application_reference NVARCHAR(100) NOT NULL,
        customer_name NVARCHAR(255) NULL,
        decision_type NVARCHAR(50) NOT NULL,
        product_type NVARCHAR(100) NULL,
        requested_amount FLOAT NULL,
        approved_amount FLOAT NULL,
        approved_limit FLOAT NULL,
        monthly_payment FLOAT NULL,
        net_monthly_income FLOAT NULL,
        monthly_expenses FLOAT NULL,
        existing_debt_payments FLOAT NULL,
        credit_score INT NULL,
        debt_to_income_ratio FLOAT NULL,
        fraud_score FLOAT NULL,
        fraud_flag BIT NULL,
        risk_probability FLOAT NULL,
        probability_default FLOAT NULL,
        ifrs9_stage NVARCHAR(50) NULL,
        ecl_lifetime FLOAT NULL,
        final_decision NVARCHAR(50) NULL,
        llm_explanation NVARCHAR(MAX) NULL,
        created_at DATETIME NOT NULL DEFAULT GETDATE()
    );
    """

    summary_view_sql = """
    CREATE OR ALTER VIEW analytics.v_portfolio_summary AS
    SELECT
        COUNT(*) AS total_applications,
        SUM(CASE WHEN final_decision = 'APPROVED' THEN 1 ELSE 0 END) AS approved_cases,
        CAST(
            1.0 * SUM(CASE WHEN final_decision = 'APPROVED' THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0)
            AS FLOAT
        ) AS approval_rate,
        SUM(ISNULL(approved_amount, 0)) AS loan_exposure,
        SUM(ISNULL(approved_limit, 0)) AS credit_limits,
        SUM(ISNULL(ecl_lifetime, 0)) AS lifetime_ecl,
        AVG(ISNULL(probability_default, 0)) AS average_pd,
        AVG(ISNULL(fraud_score, 0)) AS average_fraud_score,
        AVG(ISNULL(risk_probability, 0)) AS average_shap_risk,
        SUM(CASE WHEN ISNULL(fraud_score, 0) >= 0.80 THEN 1 ELSE 0 END) AS critical_alerts,
        SUM(CASE WHEN ISNULL(fraud_score, 0) >= 0.50 AND ISNULL(fraud_score, 0) < 0.80 THEN 1 ELSE 0 END) AS high_alerts
    FROM analytics.fact_applications;
    """

    recent_view_sql = """
    CREATE OR ALTER VIEW analytics.v_recent_applications AS
    SELECT TOP 100
        application_reference,
        customer_name,
        decision_type,
        product_type,
        requested_amount,
        approved_amount,
        approved_limit,
        monthly_payment,
        net_monthly_income,
        existing_debt_payments,
        credit_score,
        debt_to_income_ratio,
        fraud_score,
        fraud_flag,
        risk_probability,
        probability_default,
        ifrs9_stage,
        ecl_lifetime,
        final_decision,
        llm_explanation,
        created_at
    FROM analytics.fact_applications
    ORDER BY created_at DESC;
    """

    product_view_sql = """
    CREATE OR ALTER VIEW analytics.v_product_distribution AS
    SELECT
        product_type AS product,
        COUNT(*) AS [count]
    FROM analytics.fact_applications
    GROUP BY product_type;
    """

    decision_view_sql = """
    CREATE OR ALTER VIEW analytics.v_decision_distribution AS
    SELECT
        final_decision AS decision,
        COUNT(*) AS [count]
    FROM analytics.fact_applications
    GROUP BY final_decision;
    """

    fraud_view_sql = """
    CREATE OR ALTER VIEW analytics.v_fraud_distribution AS
    SELECT
        CASE
            WHEN ISNULL(fraud_score, 0) >= 0.80 THEN 'Critical'
            WHEN ISNULL(fraud_score, 0) >= 0.50 THEN 'High'
            WHEN ISNULL(fraud_score, 0) >= 0.20 THEN 'Medium'
            ELSE 'Low'
        END AS alert_level,
        COUNT(*) AS [count]
    FROM analytics.fact_applications
    GROUP BY
        CASE
            WHEN ISNULL(fraud_score, 0) >= 0.80 THEN 'Critical'
            WHEN ISNULL(fraud_score, 0) >= 0.50 THEN 'High'
            WHEN ISNULL(fraud_score, 0) >= 0.20 THEN 'Medium'
            ELSE 'Low'
        END;
    """

    with engine.begin() as conn:
        conn.execute(text(schema_sql))
        conn.execute(text(prediction_log_sql))
        conn.execute(text(fact_sql))
        conn.execute(text(summary_view_sql))
        conn.execute(text(recent_view_sql))
        conn.execute(text(product_view_sql))
        conn.execute(text(decision_view_sql))
        conn.execute(text(fraud_view_sql))


init_schema()


# =========================================================
# REQUEST MODELS
# =========================================================
class LoanRequest(BaseModel):
    customer_name: str = Field(..., min_length=1)
    product_type: str = "personal_loan"
    requested_amount: float
    annual_interest_rate: float = 15.5
    term_months: int = 60
    net_monthly_income: float
    monthly_expenses: float = 0.0
    existing_debt_payments: float = 0.0
    credit_score: int
    fraud_score: float = 0.05
    property_value: float = 0.0
    deposit: float = 0.0
    secured: bool = False
    days_past_due: int = 0
    sicr_flag: bool = False
    default_flag: bool = False
    affordability_factor: float = 0.70
    debt_to_income_cap: float = 0.45
    stress_rate_addon: float = 2.0


class CreditRequest(BaseModel):
    customer_name: str = Field(..., min_length=1)
    product_type: str = "credit_card"
    net_monthly_income: float
    existing_debt_payments: float = 0.0
    credit_score: int
    fraud_score: float = 0.05
    days_past_due: int = 0
    sicr_flag: bool = False
    default_flag: bool = False


# =========================================================
# SCORING ENGINES
# =========================================================
def assess_loan(req: LoanRequest) -> Dict[str, Any]:
    requested_amount = to_float(req.requested_amount)
    income = to_float(req.net_monthly_income)
    expenses = to_float(req.monthly_expenses)
    debt = to_float(req.existing_debt_payments)
    annual_interest_rate = to_float(req.annual_interest_rate)
    term_months = max(1, to_int(req.term_months, 60))
    credit_score = to_int(req.credit_score)
    fraud_score = to_float(req.fraud_score)
    affordability_factor = to_float(req.affordability_factor, 0.70)
    debt_to_income_cap = to_float(req.debt_to_income_cap, 0.45)
    stress_rate_addon = to_float(req.stress_rate_addon, 2.0)
    days_past_due = to_int(req.days_past_due)
    sicr_flag = to_bool(req.sicr_flag)
    default_flag = to_bool(req.default_flag)

    disposable_income = max(0.0, income - expenses - debt)
    debt_to_income_ratio = debt / income if income > 0 else 1.0

    credit_component = max(0.0, min(1.0, (700 - credit_score) / 500))
    dti_component = min(1.0, debt_to_income_ratio)
    amount_component = min(1.0, requested_amount / max(income * 12, 1))
    fraud_component = min(1.0, fraud_score)

    risk_probability = round(
        min(
            0.95,
            max(
                0.02,
                0.45 * credit_component
                + 0.25 * dti_component
                + 0.15 * amount_component
                + 0.15 * fraud_component,
            ),
        ),
        4,
    )
    probability_default = risk_probability

    monthly_rate = ((annual_interest_rate + stress_rate_addon) / 100) / 12
    if monthly_rate > 0:
        monthly_payment = requested_amount * (
            (monthly_rate * (1 + monthly_rate) ** term_months)
            / (((1 + monthly_rate) ** term_months) - 1)
        )
    else:
        monthly_payment = requested_amount / term_months

    max_affordable_payment = disposable_income * affordability_factor
    stage = ifrs9_stage(days_past_due, sicr_flag, default_flag)

    if fraud_score >= 0.80:
        final_decision = "DECLINED"
    elif default_flag:
        final_decision = "DECLINED"
    elif debt_to_income_ratio > debt_to_income_cap:
        final_decision = "DECLINED"
    elif monthly_payment > max_affordable_payment:
        final_decision = "DECLINED"
    elif risk_probability >= 0.55:
        final_decision = "DECLINED"
    elif risk_probability >= 0.35:
        final_decision = "REVIEW"
    else:
        final_decision = "APPROVED"

    approved_amount = requested_amount if final_decision == "APPROVED" else 0.0
    ecl_lifetime = round(probability_default * max(requested_amount, 1) * 0.12, 2)

    shap_features = build_shap_features(
        credit_score=credit_score,
        debt_to_income_ratio=debt_to_income_ratio,
        income=income,
        fraud_score=fraud_score,
        amount=requested_amount,
    )

    llm_explanation = safe_llm_explanation(
        credit_score=credit_score,
        income=income,
        debt=debt,
        decision=final_decision,
        risk=risk_probability,
        dti=debt_to_income_ratio,
        fraud_score=fraud_score,
        stage=stage,
        decision_type="loan",
        shap_features=shap_features,
    )

    return {
        "application_reference": make_reference("LOAN"),
        "customer_name": req.customer_name,
        "decision_type": "loan",
        "product_type": req.product_type,
        "requested_amount": round(requested_amount, 2),
        "approved_amount": round(approved_amount, 2),
        "approved_limit": None,
        "monthly_payment": round(monthly_payment, 2),
        "net_monthly_income": round(income, 2),
        "monthly_expenses": round(expenses, 2),
        "existing_debt_payments": round(debt, 2),
        "credit_score": credit_score,
        "debt_to_income_ratio": round(debt_to_income_ratio, 4),
        "fraud_score": round(fraud_score, 4),
        "fraud_flag": 1 if fraud_score >= 0.50 else 0,
        "risk_probability": risk_probability,
        "probability_default": probability_default,
        "ifrs9_stage": stage,
        "ecl_lifetime": ecl_lifetime,
        "final_decision": final_decision,
        "llm_explanation": llm_explanation,
        "top_shap_features": shap_features,
        "alert_level": fraud_alert_from_score(fraud_score),
        "created_at": datetime.utcnow().isoformat(),
    }


def assess_credit(req: CreditRequest) -> Dict[str, Any]:
    income = to_float(req.net_monthly_income)
    debt = to_float(req.existing_debt_payments)
    credit_score = to_int(req.credit_score)
    fraud_score = to_float(req.fraud_score)
    days_past_due = to_int(req.days_past_due)
    sicr_flag = to_bool(req.sicr_flag)
    default_flag = to_bool(req.default_flag)

    debt_to_income_ratio = debt / income if income > 0 else 1.0
    credit_component = max(0.0, min(1.0, (700 - credit_score) / 500))
    dti_component = min(1.0, debt_to_income_ratio)
    fraud_component = min(1.0, fraud_score)

    risk_probability = round(
        min(
            0.95,
            max(
                0.02,
                0.55 * credit_component
                + 0.25 * dti_component
                + 0.20 * fraud_component,
            ),
        ),
        4,
    )
    probability_default = risk_probability
    stage = ifrs9_stage(days_past_due, sicr_flag, default_flag)

    if fraud_score >= 0.80:
        final_decision = "DECLINED"
    elif default_flag:
        final_decision = "DECLINED"
    elif risk_probability >= 0.55:
        final_decision = "DECLINED"
    elif risk_probability >= 0.35:
        final_decision = "REVIEW"
    else:
        final_decision = "APPROVED"

    base_limit = max(0.0, income * 4.5)
    approved_limit = round(base_limit if final_decision == "APPROVED" else 0.0, 2)
    ecl_lifetime = round(probability_default * max(base_limit, 1) * 0.08, 2)

    shap_features = build_shap_features(
        credit_score=credit_score,
        debt_to_income_ratio=debt_to_income_ratio,
        income=income,
        fraud_score=fraud_score,
        amount=approved_limit,
    )

    llm_explanation = safe_llm_explanation(
        credit_score=credit_score,
        income=income,
        debt=debt,
        decision=final_decision,
        risk=risk_probability,
        dti=debt_to_income_ratio,
        fraud_score=fraud_score,
        stage=stage,
        decision_type="credit",
        shap_features=shap_features,
    )

    return {
        "application_reference": make_reference("CREDIT"),
        "customer_name": req.customer_name,
        "decision_type": "credit",
        "product_type": req.product_type,
        "requested_amount": None,
        "approved_amount": None,
        "approved_limit": approved_limit,
        "monthly_payment": None,
        "net_monthly_income": round(income, 2),
        "monthly_expenses": None,
        "existing_debt_payments": round(debt, 2),
        "credit_score": credit_score,
        "debt_to_income_ratio": round(debt_to_income_ratio, 4),
        "fraud_score": round(fraud_score, 4),
        "fraud_flag": 1 if fraud_score >= 0.50 else 0,
        "risk_probability": risk_probability,
        "probability_default": probability_default,
        "ifrs9_stage": stage,
        "ecl_lifetime": ecl_lifetime,
        "final_decision": final_decision,
        "llm_explanation": llm_explanation,
        "top_shap_features": shap_features,
        "alert_level": fraud_alert_from_score(fraud_score),
        "created_at": datetime.utcnow().isoformat(),
    }


# =========================================================
# DATABASE SAVE
# =========================================================
def save_application(result: Dict[str, Any]) -> None:
    db_row = {k: v for k, v in result.items() if k not in {"top_shap_features", "alert_level", "created_at"}}

    insert_log_sql = """
    INSERT INTO ml.prediction_log (
        application_reference,
        customer_name,
        decision_type,
        product_type,
        requested_amount,
        approved_amount,
        approved_limit,
        monthly_payment,
        net_monthly_income,
        monthly_expenses,
        existing_debt_payments,
        credit_score,
        debt_to_income_ratio,
        fraud_score,
        fraud_flag,
        risk_probability,
        probability_default,
        ifrs9_stage,
        ecl_lifetime,
        final_decision,
        llm_explanation,
        created_at
    )
    VALUES (
        :application_reference,
        :customer_name,
        :decision_type,
        :product_type,
        :requested_amount,
        :approved_amount,
        :approved_limit,
        :monthly_payment,
        :net_monthly_income,
        :monthly_expenses,
        :existing_debt_payments,
        :credit_score,
        :debt_to_income_ratio,
        :fraud_score,
        :fraud_flag,
        :risk_probability,
        :probability_default,
        :ifrs9_stage,
        :ecl_lifetime,
        :final_decision,
        :llm_explanation,
        GETDATE()
    )
    """

    insert_fact_sql = """
    INSERT INTO analytics.fact_applications (
        application_reference,
        customer_name,
        decision_type,
        product_type,
        requested_amount,
        approved_amount,
        approved_limit,
        monthly_payment,
        net_monthly_income,
        monthly_expenses,
        existing_debt_payments,
        credit_score,
        debt_to_income_ratio,
        fraud_score,
        fraud_flag,
        risk_probability,
        probability_default,
        ifrs9_stage,
        ecl_lifetime,
        final_decision,
        llm_explanation,
        created_at
    )
    VALUES (
        :application_reference,
        :customer_name,
        :decision_type,
        :product_type,
        :requested_amount,
        :approved_amount,
        :approved_limit,
        :monthly_payment,
        :net_monthly_income,
        :monthly_expenses,
        :existing_debt_payments,
        :credit_score,
        :debt_to_income_ratio,
        :fraud_score,
        :fraud_flag,
        :risk_probability,
        :probability_default,
        :ifrs9_stage,
        :ecl_lifetime,
        :final_decision,
        :llm_explanation,
        GETDATE()
    )
    """

    with engine.begin() as conn:
        conn.execute(text(insert_log_sql), db_row)
        conn.execute(text(insert_fact_sql), db_row)


# =========================================================
# BASIC ROUTES
# =========================================================
@app.get("/")
def home():
    return {"message": "Full Fintech Banking Platform API is running"}


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}


# =========================================================
# REQUEST ROUTES
# =========================================================
@app.post("/api/loan/assess")
def loan_assess(req: LoanRequest):
    result = assess_loan(req)
    save_application(result)
    return result


@app.post("/api/credit/assess")
def credit_assess(req: CreditRequest):
    result = assess_credit(req)
    save_application(result)
    return result


# =========================================================
# DASHBOARD ROUTES
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM analytics.v_portfolio_summary")).mappings().first()
        return dict(row) if row else {}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/recent-loans")
def recent_loans():
    sql = """
    SELECT TOP 20
        application_reference,
        customer_name,
        requested_amount,
        approved_amount,
        fraud_flag,
        product_type,
        final_decision,
        created_at
    FROM analytics.fact_applications
    WHERE decision_type = 'loan'
    ORDER BY created_at DESC
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/recent-credit")
def recent_credit():
    sql = """
    SELECT TOP 20
        application_reference,
        customer_name,
        approved_limit,
        fraud_flag,
        product_type,
        final_decision,
        created_at
    FROM analytics.fact_applications
    WHERE decision_type = 'credit'
    ORDER BY created_at DESC
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/product-distribution")
def product_distribution():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM analytics.v_product_distribution")).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/decision-distribution")
def decision_distribution():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM analytics.v_decision_distribution")).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/fraud-distribution")
def fraud_distribution():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM analytics.v_fraud_distribution")).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/fraud/live")
def fraud_live():
    sql = """
    SELECT TOP 100
        application_reference AS TransactionID,
        ISNULL(requested_amount, approved_limit) AS TransactionAmt,
        fraud_flag AS isFraud,
        created_at AS event_time,
        CASE
            WHEN ISNULL(fraud_score, 0) >= 0.80 THEN 'Critical'
            WHEN ISNULL(fraud_score, 0) >= 0.50 THEN 'High'
            WHEN ISNULL(fraud_score, 0) >= 0.20 THEN 'Medium'
            ELSE 'Low'
        END AS alert_level,
        fraud_score
    FROM analytics.fact_applications
    ORDER BY created_at DESC
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}