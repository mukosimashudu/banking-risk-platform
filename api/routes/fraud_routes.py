from fastapi import APIRouter
from sqlalchemy import text

from src.config.db import engine


router = APIRouter(prefix="/api/fraud", tags=["Fraud"])


@router.get("/live")
def fraud_live():
    if engine is None:
        return []

    sql = """
    SELECT TOP 100
        application_reference AS TransactionID,
        ISNULL(NULLIF(requested_amount, 0), approved_limit) AS TransactionAmt,
        CASE
            WHEN ISNULL(fraud_score, 0) >= 0.75 THEN 1
            ELSE 0
        END AS isFraud,
        created_at AS event_time,
        CASE
            WHEN ISNULL(fraud_score, 0) >= 0.90 THEN 'Critical'
            WHEN ISNULL(fraud_score, 0) >= 0.75 THEN 'High'
            WHEN ISNULL(fraud_score, 0) >= 0.45 THEN 'Medium'
            ELSE 'Low'
        END AS alert_level,
        fraud_score,
        customer_name,
        decision_type,
        final_decision
    FROM ml.prediction_log
    ORDER BY created_at DESC
    """

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()

        return [dict(r) for r in rows]

    except Exception as exc:
        return {"error": str(exc)}