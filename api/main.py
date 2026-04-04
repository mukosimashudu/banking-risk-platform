from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from src.config.db import engine

app = FastAPI(title="Banking Risk Platform API")


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"message": "Full Fintech Banking Platform API is running"}


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# EXECUTIVE SUMMARY (SAFE)
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

            avg_amount = conn.execute(text("""
                SELECT AVG(TransactionAmt) FROM dbo.train_transaction
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
        print("🔥 ERROR SUMMARY:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# RECENT TRANSACTIONS
# =========================
@app.get("/api/portfolio/recent")
def get_recent():
    try:
        query = text("""
            SELECT TOP 50
                TransactionID AS application_reference,
                TransactionAmt AS requested_amount,
                isFraud AS fraud_flag,
                ProductCD AS product_type
            FROM dbo.train_transaction
            ORDER BY TransactionID DESC
        """)

        with engine.connect() as conn:
            result = conn.execute(query)
            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 ERROR RECENT:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# LIVE FRAUD MONITORING
# =========================
@app.get("/api/fraud/recent")
def fraud_monitor():
    try:
        query = text("""
            SELECT TOP 100
                TransactionID,
                TransactionAmt,
                isFraud,
                GETDATE() AS event_time,
                CASE 
                    WHEN isFraud = 1 THEN 'Critical'
                    ELSE 'Low'
                END AS alert_level,
                TransactionAmt / 1000.0 AS fraud_score
            FROM dbo.train_transaction
            ORDER BY TransactionID DESC
        """)

        with engine.connect() as conn:
            result = conn.execute(query)
            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 ERROR FRAUD:", str(e))
        raise HTTPException(status_code=500, detail=str(e))