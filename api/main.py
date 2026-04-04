from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from src.llm.llm_assistant import generate_explanation
from src.scoring.decision_engine import make_credit_card_decision, make_final_decision
from src.scoring.explainability import explain_application
from src.scoring.fraud_monitor import build_fraud_event
from src.scoring.ifrs9_engine import ifrs9_engine
from src.scoring.loan_engine import affordability_engine, build_amortisation_schedule


app = FastAPI(
    title="Full Fintech Banking Platform API",
    version="5.0.0",
    description="Loans, credit, IFRS 9, explainable AI, fraud monitoring, and executive dashboard APIs.",
)

metadata = MetaData()

loan_applications = Table(
    "loan_applications",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("application_reference", String(100), nullable=False),
    Column("customer_name", String(200), nullable=False),
    Column("product_type", String(50), nullable=False),
    Column("requested_amount", Float, nullable=False),
    Column("recommended_amount", Float, nullable=False),
    Column("approved_amount", Float, nullable=False),
    Column("annual_interest_rate", Float, nullable=False),
    Column("term_months", Integer, nullable=False),
    Column("monthly_payment", Float, nullable=False),
    Column("total_interest", Float, nullable=False),
    Column("total_repayment", Float, nullable=False),
    Column("net_monthly_income", Float, nullable=False),
    Column("monthly_expenses", Float, nullable=False),
    Column("existing_debt_payments", Float, nullable=False),
    Column("disposable_income", Float, nullable=False),
    Column("debt_to_income_ratio", Float, nullable=False),
    Column("expense_to_income_ratio", Float, nullable=False),
    Column("max_affordable_payment", Float, nullable=False),
    Column("stressed_monthly_payment", Float, nullable=False),
    Column("property_value", Float, nullable=True),
    Column("deposit", Float, nullable=True),
    Column("ltv", Float, nullable=True),
    Column("credit_score", Integer, nullable=False),
    Column("fraud_score", Float, nullable=False),
    Column("secured", Boolean, nullable=False),
    Column("days_past_due", Integer, nullable=False),
    Column("sicr_flag", Boolean, nullable=False),
    Column("default_flag", Boolean, nullable=False),
    Column("ifrs9_stage", String(20), nullable=False),
    Column("pd_12m", Float, nullable=False),
    Column("pd_lifetime", Float, nullable=False),
    Column("lgd", Float, nullable=False),
    Column("ead", Float, nullable=False),
    Column("ecl_12m", Float, nullable=False),
    Column("ecl_lifetime", Float, nullable=False),
    Column("shap_risk_probability", Float, nullable=True),
    Column("llm_explanation", String(4000), nullable=True),
    Column("final_decision", String(50), nullable=False),
    Column("decision_reason", String(500), nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

amortisation_schedule = Table(
    "amortisation_schedule",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("application_reference", String(100), nullable=False),
    Column("instalment_no", Integer, nullable=False),
    Column("opening_balance", Float, nullable=False),
    Column("instalment", Float, nullable=False),
    Column("principal_component", Float, nullable=False),
    Column("interest_component", Float, nullable=False),
    Column("closing_balance", Float, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

credit_applications = Table(
    "credit_applications",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("application_reference", String(100), nullable=False),
    Column("customer_name", String(200), nullable=False),
    Column("product_type", String(50), nullable=False),
    Column("net_monthly_income", Float, nullable=False),
    Column("existing_debt_payments", Float, nullable=False),
    Column("credit_score", Integer, nullable=False),
    Column("risk_probability", Float, nullable=False),
    Column("approved_limit", Float, nullable=False),
    Column("final_decision", String(50), nullable=False),
    Column("decision_reason", String(500), nullable=False),
    Column("llm_explanation", String(4000), nullable=True),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

fraud_events = Table(
    "fraud_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_time", DateTime, nullable=False, default=datetime.utcnow),
    Column("application_reference", String(100), nullable=False),
    Column("customer_name", String(200), nullable=False),
    Column("product_type", String(50), nullable=False),
    Column("requested_amount", Float, nullable=False),
    Column("fraud_score", Float, nullable=False),
    Column("alert_level", String(20), nullable=False),
    Column("final_decision", String(50), nullable=False),
    Column("message", String(500), nullable=False),
)


def get_db_url() -> str:
    db_server = os.getenv("DB_SERVER")
    db_database = os.getenv("DB_DATABASE")
    db_username = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_driver = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")

    if all([db_server, db_database, db_username, db_password]):
        driver = db_driver.replace(" ", "+")
        return (
            f"mssql+pyodbc://{db_username}:{db_password}@{db_server}/{db_database}"
            f"?driver={driver}&Encrypt=yes&TrustServerCertificate=yes"
        )

    return "sqlite:///banking_risk_platform.db"


def get_engine() -> Engine:
    return create_engine(get_db_url(), future=True)


engine = get_engine()


def init_db() -> None:
    metadata.create_all(engine)


@app.on_event("startup")
def startup_event() -> None:
    init_db()


class LoanAssessmentRequest(BaseModel):
    customer_name: str = Field(..., min_length=2)
    product_type: str = Field(..., description="personal_loan, home_loan, vehicle_loan, credit_card")
    requested_amount: float = Field(..., gt=0)
    annual_interest_rate: float = Field(..., ge=0)
    term_months: int = Field(..., gt=0)
    net_monthly_income: float = Field(..., ge=0)
    monthly_expenses: float = Field(..., ge=0)
    existing_debt_payments: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=900)
    fraud_score: float = Field(0.05, ge=0, le=1)
    property_value: Optional[float] = Field(None, ge=0)
    deposit: Optional[float] = Field(None, ge=0)
    secured: bool = False
    days_past_due: int = Field(0, ge=0)
    sicr_flag: bool = False
    default_flag: bool = False
    affordability_factor: float = Field(0.70, gt=0, le=1)
    debt_to_income_cap: float = Field(0.45, gt=0, le=1)
    stress_rate_addon: float = Field(2.00, ge=0)


class CreditAssessmentRequest(BaseModel):
    customer_name: str = Field(..., min_length=2)
    product_type: str = Field(default="credit_card")
    net_monthly_income: float = Field(..., ge=0)
    existing_debt_payments: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=900)


def build_reference(product_type: str) -> str:
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    prefix = (product_type or "APP").upper()[:4]
    return f"{prefix}-{now}"


def load_credit_model():
    model_path = os.path.join("models", "credit_model_best.pkl")
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def credit_model_probability(payload: CreditAssessmentRequest) -> float:
    model = load_credit_model()
    if model is None:
        base = 0.50
        if payload.credit_score >= 720:
            base -= 0.20
        elif payload.credit_score >= 660:
            base -= 0.10
        elif payload.credit_score < 580:
            base += 0.15

        dti = payload.existing_debt_payments / payload.net_monthly_income if payload.net_monthly_income > 0 else 1.0
        base += max(0.0, dti - 0.20)
        return max(0.0, min(1.0, base))

    try:
        df = pd.DataFrame(
            [
                {
                    "net_monthly_income": payload.net_monthly_income,
                    "monthly_expenses": 0.0,
                    "existing_debt_payments": payload.existing_debt_payments,
                    "requested_amount": payload.net_monthly_income * 2.0,
                    "annual_interest_rate": 0.0,
                    "term_months": 12,
                    "credit_score": payload.credit_score,
                    "fraud_score": 0.05,
                    "property_value": 0.0,
                    "deposit": 0.0,
                    "days_past_due": 0,
                    "sicr_flag": 0,
                    "default_flag": 0,
                }
            ]
        )
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            if len(proba) > 1:
                return float(proba[1])
        pred = model.predict(df)
        return float(pred[0])
    except Exception:
        return 0.50


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Full Fintech Banking Platform API is running"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "database_backend": get_db_url().split("://")[0]}


@app.post("/api/schema/init")
def initialise_schema() -> Dict[str, str]:
    init_db()
    return {"message": "Database schema created successfully"}


@app.post("/api/loan/assess")
def assess_loan(payload: LoanAssessmentRequest):
    try:
        result = affordability_engine(
            net_monthly_income=payload.net_monthly_income,
            monthly_expenses=payload.monthly_expenses,
            existing_debt_payments=payload.existing_debt_payments,
            requested_amount=payload.requested_amount,
            annual_interest_rate=payload.annual_interest_rate,
            term_months=payload.term_months,
            affordability_factor=payload.affordability_factor,
            debt_to_income_cap=payload.debt_to_income_cap,
            stress_rate_addon=payload.stress_rate_addon,
            property_value=payload.property_value,
            deposit=payload.deposit,
        )

        ifrs9 = ifrs9_engine(
            approved_amount=result.approved_amount,
            credit_score=payload.credit_score,
            product_type=payload.product_type,
            secured=payload.secured,
            days_past_due=payload.days_past_due,
            significant_increase_in_credit_risk=payload.sicr_flag,
            default_flag=payload.default_flag,
            ltv=result.ltv,
            ccf=1.0,
        )

        final_decision, reason = make_final_decision(
            product_type=payload.product_type,
            requested_amount=payload.requested_amount,
            approved_amount=result.approved_amount,
            affordability_pass=result.affordability_pass,
            fraud_score=payload.fraud_score,
            credit_score=payload.credit_score,
            ifrs9_stage=ifrs9.stage,
            debt_to_income_ratio=result.debt_to_income_ratio,
            ltv=result.ltv,
        )

        schedule = build_amortisation_schedule(
            principal=result.approved_amount,
            annual_interest_rate=payload.annual_interest_rate,
            term_months=payload.term_months,
        )

        shap_explanation = explain_application(payload.model_dump())

        llm_text = generate_explanation(
            {
                "product_type": payload.product_type,
                "income": payload.net_monthly_income,
                "expenses": payload.monthly_expenses,
                "debt": payload.existing_debt_payments,
                "credit_score": payload.credit_score,
                "fraud_score": payload.fraud_score,
                "requested": result.requested_amount,
                "approved": result.approved_amount,
                "monthly_payment": result.monthly_payment,
                "pd": ifrs9.pd_12m,
                "lgd": ifrs9.lgd,
                "ecl": ifrs9.ecl_lifetime,
                "stage": ifrs9.stage,
                "decision": final_decision,
                "reason": reason,
            }
        )

        application_reference = build_reference(payload.product_type)

        fraud_event = build_fraud_event(
            application_reference=application_reference,
            customer_name=payload.customer_name,
            product_type=payload.product_type,
            requested_amount=payload.requested_amount,
            fraud_score=payload.fraud_score,
            final_decision=final_decision,
        )

        with engine.begin() as conn:
            conn.execute(
                insert(loan_applications).values(
                    application_reference=application_reference,
                    customer_name=payload.customer_name,
                    product_type=payload.product_type,
                    requested_amount=result.requested_amount,
                    recommended_amount=result.recommended_amount,
                    approved_amount=result.approved_amount,
                    annual_interest_rate=payload.annual_interest_rate,
                    term_months=payload.term_months,
                    monthly_payment=result.monthly_payment,
                    total_interest=result.total_interest,
                    total_repayment=result.total_repayment,
                    net_monthly_income=payload.net_monthly_income,
                    monthly_expenses=payload.monthly_expenses,
                    existing_debt_payments=payload.existing_debt_payments,
                    disposable_income=result.disposable_income,
                    debt_to_income_ratio=result.debt_to_income_ratio,
                    expense_to_income_ratio=result.expense_to_income_ratio,
                    max_affordable_payment=result.max_affordable_payment,
                    stressed_monthly_payment=result.stressed_monthly_payment,
                    property_value=payload.property_value,
                    deposit=payload.deposit,
                    ltv=result.ltv,
                    credit_score=payload.credit_score,
                    fraud_score=payload.fraud_score,
                    secured=payload.secured,
                    days_past_due=payload.days_past_due,
                    sicr_flag=payload.sicr_flag,
                    default_flag=payload.default_flag,
                    ifrs9_stage=ifrs9.stage,
                    pd_12m=ifrs9.pd_12m,
                    pd_lifetime=ifrs9.pd_lifetime,
                    lgd=ifrs9.lgd,
                    ead=ifrs9.ead,
                    ecl_12m=ifrs9.ecl_12m,
                    ecl_lifetime=ifrs9.ecl_lifetime,
                    shap_risk_probability=shap_explanation.get("risk_probability"),
                    llm_explanation=llm_text[:4000],
                    final_decision=final_decision,
                    decision_reason=reason,
                    created_at=datetime.utcnow(),
                )
            )

            if schedule:
                conn.execute(
                    insert(amortisation_schedule),
                    [
                        {
                            "application_reference": application_reference,
                            "instalment_no": row["instalment_no"],
                            "opening_balance": row["opening_balance"],
                            "instalment": row["instalment"],
                            "principal_component": row["principal_component"],
                            "interest_component": row["interest_component"],
                            "closing_balance": row["closing_balance"],
                            "created_at": datetime.utcnow(),
                        }
                        for row in schedule
                    ],
                )

            conn.execute(
                insert(fraud_events).values(
                    event_time=datetime.utcnow(),
                    application_reference=fraud_event["application_reference"],
                    customer_name=fraud_event["customer_name"],
                    product_type=fraud_event["product_type"],
                    requested_amount=fraud_event["requested_amount"],
                    fraud_score=fraud_event["fraud_score"],
                    alert_level=fraud_event["alert_level"],
                    final_decision=fraud_event["final_decision"],
                    message=fraud_event["message"],
                )
            )

        return {
            "application_reference": application_reference,
            "customer_name": payload.customer_name,
            "product_type": payload.product_type,
            "final_decision": final_decision,
            "decision_reason": reason,
            "requested_amount": result.requested_amount,
            "recommended_amount": result.recommended_amount,
            "approved_amount": result.approved_amount,
            "annual_interest_rate": payload.annual_interest_rate,
            "term_months": payload.term_months,
            "monthly_payment": result.monthly_payment,
            "total_interest": result.total_interest,
            "total_repayment": result.total_repayment,
            "affordability_pass": result.affordability_pass,
            "debt_to_income_ratio": result.debt_to_income_ratio,
            "expense_to_income_ratio": result.expense_to_income_ratio,
            "disposable_income": result.disposable_income,
            "max_affordable_payment": result.max_affordable_payment,
            "stressed_monthly_payment": result.stressed_monthly_payment,
            "ltv": result.ltv,
            "ifrs9_stage": ifrs9.stage,
            "pd_12m": ifrs9.pd_12m,
            "pd_lifetime": ifrs9.pd_lifetime,
            "lgd": ifrs9.lgd,
            "ead": ifrs9.ead,
            "ecl_12m": ifrs9.ecl_12m,
            "ecl_lifetime": ifrs9.ecl_lifetime,
            "shap_explanation": shap_explanation,
            "llm_explanation": llm_text,
            "fraud_event": fraud_event,
            "amortisation_schedule": schedule,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/credit/assess")
def assess_credit(payload: CreditAssessmentRequest):
    try:
        risk_probability = credit_model_probability(payload)
        decision, approved_limit, reason = make_credit_card_decision(
            income=payload.net_monthly_income,
            debt=payload.existing_debt_payments,
            credit_score=payload.credit_score,
            risk_probability=risk_probability,
        )

        application_reference = build_reference(payload.product_type)

        explanation = generate_explanation(
            {
                "product_type": payload.product_type,
                "income": payload.net_monthly_income,
                "expenses": 0.0,
                "debt": payload.existing_debt_payments,
                "credit_score": payload.credit_score,
                "fraud_score": 0.0,
                "requested": approved_limit,
                "approved": approved_limit,
                "monthly_payment": 0.0,
                "pd": risk_probability,
                "lgd": 0.65,
                "ecl": approved_limit * risk_probability * 0.65,
                "stage": "Stage 1",
                "decision": decision,
                "reason": reason,
            }
        )

        with engine.begin() as conn:
            conn.execute(
                insert(credit_applications).values(
                    application_reference=application_reference,
                    customer_name=payload.customer_name,
                    product_type=payload.product_type,
                    net_monthly_income=payload.net_monthly_income,
                    existing_debt_payments=payload.existing_debt_payments,
                    credit_score=payload.credit_score,
                    risk_probability=risk_probability,
                    approved_limit=approved_limit,
                    final_decision=decision,
                    decision_reason=reason,
                    llm_explanation=explanation[:4000],
                    created_at=datetime.utcnow(),
                )
            )

        return {
            "application_reference": application_reference,
            "customer_name": payload.customer_name,
            "product_type": payload.product_type,
            "final_decision": decision,
            "decision_reason": reason,
            "risk_probability": round(risk_probability, 6),
            "approved_limit": round(float(approved_limit), 2),
            "llm_explanation": explanation,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/portfolio/recent")
def recent_applications(limit: int = 50):
    try:
        stmt = (
            select(
                loan_applications.c.application_reference,
                loan_applications.c.customer_name,
                loan_applications.c.product_type,
                loan_applications.c.requested_amount,
                loan_applications.c.approved_amount,
                loan_applications.c.monthly_payment,
                loan_applications.c.ifrs9_stage,
                loan_applications.c.ecl_lifetime,
                loan_applications.c.shap_risk_probability,
                loan_applications.c.final_decision,
                loan_applications.c.created_at,
            )
            .order_by(loan_applications.c.id.desc())
            .limit(limit)
        )

        with engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()

        return [dict(row) for row in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/credit/recent")
def recent_credit_applications(limit: int = 50):
    try:
        stmt = (
            select(
                credit_applications.c.application_reference,
                credit_applications.c.customer_name,
                credit_applications.c.product_type,
                credit_applications.c.risk_probability,
                credit_applications.c.approved_limit,
                credit_applications.c.final_decision,
                credit_applications.c.created_at,
            )
            .order_by(credit_applications.c.id.desc())
            .limit(limit)
        )

        with engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()

        return [dict(row) for row in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/fraud/recent")
def recent_fraud_events(limit: int = 100):
    try:
        stmt = (
            select(
                fraud_events.c.event_time,
                fraud_events.c.application_reference,
                fraud_events.c.customer_name,
                fraud_events.c.product_type,
                fraud_events.c.requested_amount,
                fraud_events.c.fraud_score,
                fraud_events.c.alert_level,
                fraud_events.c.final_decision,
                fraud_events.c.message,
            )
            .order_by(fraud_events.c.id.desc())
            .limit(limit)
        )

        with engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()

        return [dict(row) for row in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/portfolio/summary")
def portfolio_summary():
    try:
        with engine.begin() as conn:
            total_loan_apps = conn.execute(select(func.count()).select_from(loan_applications)).scalar() or 0
            total_credit_apps = conn.execute(select(func.count()).select_from(credit_applications)).scalar() or 0
            total_apps = int(total_loan_apps) + int(total_credit_apps)

            total_approved_loans = (
                conn.execute(
                    select(func.count()).select_from(loan_applications).where(
                        loan_applications.c.final_decision.in_(["Approve", "Approve with Reduced Amount"])
                    )
                ).scalar()
                or 0
            )

            total_approved_credit = (
                conn.execute(
                    select(func.count()).select_from(credit_applications).where(
                        credit_applications.c.final_decision.in_(["Approve", "Approve with Limit"])
                    )
                ).scalar()
                or 0
            )

            total_approved_cases = int(total_approved_loans) + int(total_approved_credit)

            total_approved_amount = (
                conn.execute(select(func.coalesce(func.sum(loan_applications.c.approved_amount), 0.0))).scalar()
                or 0.0
            )

            total_credit_limit = (
                conn.execute(select(func.coalesce(func.sum(credit_applications.c.approved_limit), 0.0))).scalar()
                or 0.0
            )

            total_ecl = (
                conn.execute(select(func.coalesce(func.sum(loan_applications.c.ecl_lifetime), 0.0))).scalar()
                or 0.0
            )

            avg_pd = (
                conn.execute(select(func.coalesce(func.avg(loan_applications.c.pd_12m), 0.0))).scalar()
                or 0.0
            )

            avg_shap_risk = (
                conn.execute
                (select(func.coalesce(func.avg(loan_applications.c.shap_risk_probability), 0.0))).scalar()
                or 0.0
            )

            avg_fraud_score = (
                conn.execute(select(func.coalesce(func.avg(fraud_events.c.fraud_score), 0.0))).scalar()
                or 0.0
            )

            critical_alerts = (
                conn.execute(
                    select(func.count()).select_from(fraud_events).where(fraud_events.c.alert_level == "Critical")
                ).scalar()
                or 0
            )

            high_alerts = (
                conn.execute(
                    select(func.count()).select_from(fraud_events).where(fraud_events.c.alert_level == "High")
                ).scalar()
                or 0
            )

            product_rows = conn.execute(
                select(
                    loan_applications.c.product_type,
                    func.count().label("count"),
                ).group_by(loan_applications.c.product_type)
            ).all()

            decision_rows = conn.execute(
                select(
                    loan_applications.c.final_decision,
                    func.count().label("count"),
                ).group_by(loan_applications.c.final_decision)
            ).all()

            fraud_rows = conn.execute(
                select(
                    fraud_events.c.alert_level,
                    func.count().label("count"),
                ).group_by(fraud_events.c.alert_level)
            ).all()

        return {
            "total_applications": total_apps,
            "total_approved_cases": total_approved_cases,
            "approval_rate": round((total_approved_cases / total_apps), 4) if total_apps else 0.0,
            "total_approved_amount": round(float(total_approved_amount), 2),
            "total_credit_limit": round(float(total_credit_limit), 2),
            "total_lifetime_ecl": round(float(total_ecl), 2),
            "average_pd_12m": round(float(avg_pd), 6),
            "average_shap_risk_probability": round(float(avg_shap_risk), 6),
            "average_fraud_score": round(float(avg_fraud_score), 6),
            "critical_alerts": int(critical_alerts),
            "high_alerts": int(high_alerts),
            "product_distribution": [{"product": r[0], "count": int(r[1])} for r in product_rows],
            "decision_distribution": [{"decision": r[0], "count": int(r[1])} for r in decision_rows],
            "fraud_distribution": [{"alert_level": r[0], "count": int(r[1])} for r in fraud_rows],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))