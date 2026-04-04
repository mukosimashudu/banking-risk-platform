from fastapi import FastAPI
from sqlalchemy import text
from src.config.db import engine

app = FastAPI(title="Banking Risk Platform API")


@app.get("/")
def root():
    return {"message": "Full Fintech Banking Platform API is running"}


# =========================
# EXECUTIVE SUMMARY
# =========================
@app.get("/api/portfolio/summary")
def get_summary():
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

    return {
        "total_applications": total or 0,
        "total_approved_cases": (total - fraud) if total else 0,
        "approval_rate": (total - fraud) / total if total else 0,
        "total_lifetime_ecl": 0,
        "total_approved_amount": avg_amount or 0,
        "total_credit_limit": 0,
        "average_pd_12m": 0.2,
        "average_fraud_score": fraud / total if total else 0,
        "critical_alerts": fraud or 0,
        "high_alerts": int((fraud or 0) * 0.5),
        "product_distribution": [],
        "decision_distribution": [],
        "fraud_distribution": [
            {"alert_level": "Fraud", "count": fraud or 0},
            {"alert_level": "Normal", "count": (total - fraud) if total else 0},
        ],
        "average_shap_risk_probability": 0.3
    }


# =========================
# RECENT APPLICATIONS
# =========================
@app.get("/api/portfolio/recent")
def get_recent():
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


# =========================
# FRAUD MONITORING (LIVE)
# =========================
@app.get("/api/fraud/recent")
def fraud_monitor():
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


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}