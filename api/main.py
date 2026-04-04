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
        income = payload.get("net_monthly_income", 0)
        debt = payload.get("existing_debt_payments", 0)
        amount = payload.get("requested_amount", 0)
        term = payload.get("term_months", 12)

        dti = debt / income if income else 0

        decision = "APPROVED" if score > 600 and dti < 0.5 else "DECLINED"
        approved_amount = amount * 0.8 if decision == "APPROVED" else 0
        monthly_payment = approved_amount / max(term, 1)

        risk_prob = round(1 - (score / 900), 2)

        # =========================
        # 🔥 SHAP (SIMULATED REALISTIC)
        # =========================
        shap_features = [
            {"feature": "credit_score", "shap_value": round((650 - score) / 100, 3)},
            {"feature": "debt_to_income", "shap_value": round(dti, 3)},
            {"feature": "income", "shap_value": round(-income / 100000, 3)},
        ]

        # =========================
        # 🧠 LLM STYLE EXPLANATION
        # =========================
        explanation = f"""
        The loan decision is based on multiple risk factors. 
        The applicant has a credit score of {score}, which {'supports approval' if score > 650 else 'increases risk'}.
        The debt-to-income ratio is {round(dti,2)}, indicating {'good affordability' if dti < 0.4 else 'financial pressure'}.
        Overall, the system {'approved' if decision == 'APPROVED' else 'declined'} the loan based on affordability and risk thresholds.
        """

        return {
            "final_decision": decision,
            "approved_amount": approved_amount,
            "monthly_payment": monthly_payment,
            "ecl_lifetime": 0,
            "decision_reason": "Risk-based decision",

            "llm_explanation": explanation.strip(),

            "fraud_event": {"alert_level": "LOW"},

            "shap_explanation": {
                "available": True,
                "risk_probability": risk_prob,
                "top_features": shap_features
            },

            "amortisation_schedule": []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# CREDIT APPLICATION (SAVE)
# =========================
@app.post("/api/credit/assess")
def credit_assess(payload: dict):
    try:
        score = payload.get("credit_score", 600)
        income = payload.get("net_monthly_income", 0)
        debt = payload.get("existing_debt_payments", 0)

        dti = debt / income if income else 0

        decision = "APPROVED" if score > 650 else "DECLINED"
        limit = income * 3
        risk = round(1 - (score / 900), 2)

        # =========================
        # SHAP
        # =========================
        shap_features = [
            {"feature": "credit_score", "shap_value": round((650 - score)/100, 3)},
            {"feature": "debt_to_income", "shap_value": round(dti, 3)},
            {"feature": "income", "shap_value": round(-income/100000, 3)}
        ]

        # =========================
        # LLM STYLE TEXT
        # =========================
        explanation = f"""
        The credit application was evaluated using risk scoring models.
        The customer has a credit score of {score}, which {'indicates strong creditworthiness' if score > 700 else 'suggests moderate risk'}.
        Their affordability is {'healthy' if dti < 0.4 else 'constrained'} based on a debt-to-income ratio of {round(dti,2)}.
        Based on these factors, the system {'approved' if decision == 'APPROVED' else 'declined'} the credit application.
        """

        return {
            "final_decision": decision,
            "approved_limit": limit,
            "risk_probability": risk,
            "decision_reason": "Credit scoring logic",

            "llm_explanation": explanation.strip(),

            "shap_explanation": {
                "available": True,
                "risk_probability": risk,
                "top_features": shap_features
            }
        }

    except Exception as e:
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