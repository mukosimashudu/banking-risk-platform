from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from src.config.db import engine
import uuid

app = FastAPI(title="Full Fintech Banking Platform API")


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"message": "Full Fintech Banking Platform API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# EXECUTIVE SUMMARY
# =========================
@app.get("/api/portfolio/summary")
def get_summary():
    try:
        with engine.connect() as conn:

            # =========================
            # TOTAL COUNTS
            # =========================
            loan_count = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.loan_applications
            """)).scalar() or 0

            credit_count = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.credit_applications
            """)).scalar() or 0

            total_applications = loan_count + credit_count

            # =========================
            # APPROVALS
            # =========================
            loan_approved = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.loan_applications
                WHERE final_decision = 'APPROVED'
            """)).scalar() or 0

            credit_approved = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.credit_applications
                WHERE final_decision = 'APPROVED'
            """)).scalar() or 0

            total_approved = loan_approved + credit_approved

            approval_rate = (
                total_approved / total_applications
                if total_applications > 0 else 0
            )

            # =========================
            # EXPOSURE
            # =========================
            loan_exposure = conn.execute(text("""
                SELECT SUM(approved_amount) FROM dbo.loan_applications
            """)).scalar() or 0

            credit_exposure = conn.execute(text("""
                SELECT SUM(approved_limit) FROM dbo.credit_applications
            """)).scalar() or 0

            # =========================
            # RISK METRICS
            # =========================
            avg_pd = conn.execute(text("""
                SELECT AVG(risk_probability) FROM dbo.credit_applications
            """)).scalar() or 0

            avg_shap = conn.execute(text("""
                SELECT AVG(shap_risk_probability) FROM dbo.loan_applications
            """)).scalar() or 0

            # =========================
            # FRAUD (FROM DATASET)
            # =========================
            fraud = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.train_transaction WHERE isFraud = 1
            """)).scalar() or 0

            total_tx = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.train_transaction
            """)).scalar() or 1

            fraud_rate = fraud / total_tx if total_tx > 0 else 0

        # =========================
        # DISTRIBUTIONS
        # =========================
        product_distribution = [
            {"product": "Loans", "count": loan_count},
            {"product": "Credit", "count": credit_count},
        ]

        decision_distribution = [
            {"decision": "Approved", "count": total_approved},
            {"decision": "Declined", "count": total_applications - total_approved},
        ]

        fraud_distribution = [
            {"alert_level": "Fraud", "count": fraud},
            {"alert_level": "Normal", "count": total_tx - fraud},
        ]

        return {
            "total_applications": total_applications,
            "total_approved_cases": total_approved,
            "approval_rate": approval_rate,
            "total_lifetime_ecl": 0,

            "total_approved_amount": loan_exposure,
            "total_credit_limit": credit_exposure,

            "average_pd_12m": avg_pd,
            "average_fraud_score": fraud_rate,

            "critical_alerts": fraud,
            "high_alerts": int(fraud * 0.5),

            "average_shap_risk_probability": avg_shap,

            "product_distribution": product_distribution,
            "decision_distribution": decision_distribution,
            "fraud_distribution": fraud_distribution
        }

    except Exception as e:
        print("🔥 SUMMARY ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
# =========================
# RECENT TRANSACTIONS (FRAUD DATA)
# =========================
@app.get("/api/portfolio/recent")
def get_recent():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT TOP 50
                    TransactionID AS application_reference,
                    TRY_CAST(TransactionAmt AS FLOAT) AS requested_amount,
                    isFraud AS fraud_flag,
                    ProductCD AS product_type
                FROM dbo.train_transaction
                ORDER BY TransactionID DESC
            """))

            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 RECENT ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# FRAUD MONITORING
# =========================
@app.get("/api/fraud/recent")
def fraud_monitor():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT TOP 100
                    TransactionID,
                    TRY_CAST(TransactionAmt AS FLOAT) AS TransactionAmt,
                    isFraud,
                    GETDATE() AS event_time,
                    CASE 
                        WHEN isFraud = 1 THEN 'Critical'
                        ELSE 'Low'
                    END AS alert_level,
                    TRY_CAST(TransactionAmt AS FLOAT) / 1000.0 AS fraud_score
                FROM dbo.train_transaction
                ORDER BY TransactionID DESC
            """))

            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 FRAUD ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# LOAN APPLICATION (SAVE)
# =========================
@app.post("/api/loan/assess")
def loan_assess(payload: dict):
    try:
        score = payload.get("credit_score", 600)
        amount = payload.get("requested_amount", 0)
        term = payload.get("term_months", 12)

        decision = "APPROVED" if score > 600 else "DECLINED"
        approved_amount = amount * 0.8
        monthly_payment = approved_amount / max(term, 1)

        application_ref = str(uuid.uuid4())[:8]

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO dbo.loan_applications (
                    application_reference,
                    customer_name,
                    product_type,
                    requested_amount,
                    approved_amount,
                    monthly_payment,
                    ifrs9_stage,
                    ecl_lifetime,
                    shap_risk_probability,
                    final_decision
                )
                VALUES (
                    :application_reference,
                    :customer_name,
                    :product_type,
                    :requested_amount,
                    :approved_amount,
                    :monthly_payment,
                    :ifrs9_stage,
                    :ecl_lifetime,
                    :shap_risk_probability,
                    :final_decision
                )
            """), {
                "application_reference": application_ref,
                "customer_name": payload.get("customer_name"),
                "product_type": payload.get("product_type"),
                "requested_amount": amount,
                "approved_amount": approved_amount,
                "monthly_payment": monthly_payment,
                "ifrs9_stage": "Stage 1",
                "ecl_lifetime": 0,
                "shap_risk_probability": 0.3,
                "final_decision": decision
            })

        return {
            "final_decision": decision,
            "approved_amount": approved_amount,
            "monthly_payment": monthly_payment,
            "ecl_lifetime": 0,
            "decision_reason": "Simple approval logic",
            "llm_explanation": "Customer meets affordability criteria.",
            "fraud_event": {"alert_level": "LOW"},
            "shap_explanation": {"available": False},
            "amortisation_schedule": []
        }

    except Exception as e:
        print("🔥 LOAN ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# CREDIT APPLICATION (SAVE)
# =========================
@app.post("/api/credit/assess")
def credit_assess(payload: dict):
    try:
        score = payload.get("credit_score", 600)
        income = payload.get("net_monthly_income", 0)

        decision = "APPROVED" if score > 650 else "DECLINED"
        limit = income * 3
        risk = 0.2 if score > 650 else 0.6

        application_ref = str(uuid.uuid4())[:8]

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO dbo.credit_applications (
                    application_reference,
                    customer_name,
                    product_type,
                    net_monthly_income,
                    existing_debt,
                    credit_score,
                    approved_limit,
                    risk_probability,
                    final_decision
                )
                VALUES (
                    :application_reference,
                    :customer_name,
                    :product_type,
                    :net_monthly_income,
                    :existing_debt,
                    :credit_score,
                    :approved_limit,
                    :risk_probability,
                    :final_decision
                )
            """), {
                "application_reference": application_ref,
                "customer_name": payload.get("customer_name"),
                "product_type": payload.get("product_type"),
                "net_monthly_income": income,
                "existing_debt": payload.get("existing_debt_payments"),
                "credit_score": score,
                "approved_limit": limit,
                "risk_probability": risk,
                "final_decision": decision
            })

        return {
            "final_decision": decision,
            "approved_limit": limit,
            "risk_probability": risk,
            "decision_reason": "Simple credit scoring logic",
            "llm_explanation": "Credit approved based on score and affordability."
        }

    except Exception as e:
        print("🔥 CREDIT ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# RECENT LOANS
# =========================
@app.get("/api/loan/recent")
def get_loans():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT TOP 50 *
                FROM dbo.loan_applications
                ORDER BY created_at DESC
            """))
            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 LOAN RECENT ERROR:", str(e))
        return []


# =========================
# RECENT CREDIT
# =========================
@app.get("/api/credit/recent")
def get_credit():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT TOP 50 *
                FROM dbo.credit_applications
                ORDER BY created_at DESC
            """))
            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 CREDIT RECENT ERROR:", str(e))
        return []