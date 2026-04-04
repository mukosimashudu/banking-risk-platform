from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def classify_alert_level(fraud_score: float) -> str:
    if fraud_score >= 0.85:
        return "Critical"
    if fraud_score >= 0.65:
        return "High"
    if fraud_score >= 0.40:
        return "Medium"
    return "Low"


def build_fraud_event(
    application_reference: str,
    customer_name: str,
    product_type: str,
    requested_amount: float,
    fraud_score: float,
    final_decision: str,
) -> Dict[str, Any]:
    level = classify_alert_level(fraud_score)

    if fraud_score >= 0.85:
        message = "Potential fraud event detected. Immediate investigation recommended."
    elif fraud_score >= 0.65:
        message = "Elevated fraud risk detected. Manual review recommended."
    elif fraud_score >= 0.40:
        message = "Moderate fraud signal detected. Monitor closely."
    else:
        message = "Fraud score within normal range."

    return {
        "event_time": datetime.utcnow().isoformat(),
        "application_reference": application_reference,
        "customer_name": customer_name,
        "product_type": product_type,
        "requested_amount": round(float(requested_amount), 2),
        "fraud_score": round(float(fraud_score), 4),
        "alert_level": level,
        "final_decision": final_decision,
        "message": message,
    }