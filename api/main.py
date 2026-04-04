from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from src.config.db import engine

app = FastAPI(title="Banking Risk Platform API")


@app.get("/")
def root():
    return {"message": "Full Fintech Banking Platform API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# EXECUTIVE SUMMARY (FIXED)
# =========================
@app.get("/api/portfolio/summary")
def get_summary():
    try:
        with engine.connect() as conn:

            total = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.train_transaction
            """)).scalar()

            fraud = conn.execute(text("""
                SELECT COUNT(*) FROM dbo.train_transaction WHERE isFraud = 1
            """)).scalar()

            # 🔥 FIX HERE (CAST TO FLOAT)
            avg_amount = conn.execute(text("""
                SELECT AVG(CAST(TransactionAmt AS FLOAT))
                FROM dbo.train_transaction
                WHERE ISNUMERIC(TransactionAmt) = 1
            """)).scalar()

        total = total or 0
        fraud = fraud or 0
        avg_amount = avg_amount or 0

        return {
            "total_applications": total,
            "total_approved_cases": total - fraud,
            "approval_rate": (total - fraud) / total if total else 0,
            "total_lifetime_ecl": 0,
            "total_approved_amount": avg_amount,
            "total_credit_limit": 0,
            "average_pd_12m": 0.2,
            "average_fraud_score": fraud / total if total else 0,
            "critical_alerts": fraud,
            "high_alerts": int(fraud * 0.5),
            "product_distribution": [],
            "decision_distribution": [],
            "fraud_distribution": [
                {"alert_level": "Fraud", "count": fraud},
                {"alert_level": "Normal", "count": total - fraud},
            ],
            "average_shap_risk_probability": 0.3
        }

    except Exception as e:
        print("🔥 SUMMARY ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# RECENT
# =========================
@app.get("/api/portfolio/recent")
def get_recent():
    try:
        query = text("""
            SELECT TOP 50
                TransactionID AS application_reference,
                TRY_CAST(TransactionAmt AS FLOAT) AS requested_amount,
                isFraud AS fraud_flag,
                ProductCD AS product_type
            FROM dbo.train_transaction
            ORDER BY TransactionID DESC
        """)

        with engine.connect() as conn:
            result = conn.execute(query)
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
        query = text("""
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
        """)

        with engine.connect() as conn:
            result = conn.execute(query)
            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 FRAUD ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# LOAN ASSESS (FIX 404)
# =========================
@app.post("/api/loan/assess")
def loan_assess(payload: dict):
    return {
        "final_decision": "APPROVED" if payload.get("credit_score", 600) > 600 else "DECLINED",
        "approved_amount": payload.get("requested_amount", 0) * 0.8,
        "monthly_payment": payload.get("requested_amount", 0) / max(payload.get("term_months", 12), 1),
        "ecl_lifetime": 0,
        "decision_reason": "Simple approval logic (demo)",
        "llm_explanation": "Customer meets affordability and risk criteria.",
        "fraud_event": {"alert_level": "LOW"},
        "shap_explanation": {"available": False},
        "amortisation_schedule": []
    }


# =========================
# CREDIT ASSESS (FIX 404)
# =========================
@app.post("/api/credit/assess")
def credit_assess(payload: dict):
    score = payload.get("credit_score", 600)

    return {
        "final_decision": "APPROVED" if score > 650 else "DECLINED",
        "approved_limit": payload.get("net_monthly_income", 0) * 3,
        "risk_probability": 0.2 if score > 650 else 0.6,
        "decision_reason": "Simple credit scoring logic",
        "llm_explanation": "Credit approved based on score and affordability."
    }