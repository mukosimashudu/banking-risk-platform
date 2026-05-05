from __future__ import annotations

from typing import Any

from src.agent.investigation_agent import run_investigation
from src.graph.graph_features import get_application_graph_risk


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def risk_decision_agent(application_reference: str) -> dict:
    result = run_investigation(application_reference)

    if "error" in result:
        return result

    data = result.get("data", {})

    graph_result = get_application_graph_risk(application_reference)
    graph_score = safe_float(graph_result.get("graph_risk_score", 0))
    graph_alert = graph_result.get("graph_alert_level", "Low")

    fraud = safe_float(data.get("fraud_score", 0))
    risk = safe_float(data.get("risk_probability", 0))

    if fraud >= 0.90 or graph_score >= 0.75:
        action = "BLOCK"
    elif fraud >= 0.75 or graph_score >= 0.50:
        action = "REVIEW"
    elif risk >= 0.60 or graph_score >= 0.25:
        action = "ESCALATE"
    else:
        action = "APPROVE"

    base_analysis = result.get("llm_analysis", "No analysis available.")
    graph_interpretation = graph_result.get("interpretation", "")

    investigation_summary = (
        f"{base_analysis} "
        f"Graph intelligence: {graph_interpretation}"
    ).strip()

    return {
        "application_reference": application_reference,
        "recommended_action": action,
        "recommendation": action,
        "analysis": investigation_summary,
        "llm_summary": base_analysis,
        "investigation_summary": investigation_summary,
        "final_decision": data.get("final_decision", "UNKNOWN"),
        "fraud_score": fraud,
        "risk_probability": risk,
        "probability_default": safe_float(data.get("probability_default", 0)),
        "credit_score": data.get("credit_score", 0),
        "transaction_amount": max(
            safe_float(data.get("requested_amount", 0)),
            safe_float(data.get("approved_amount", 0)),
            safe_float(data.get("approved_limit", 0)),
        ),
        "alert_level": data.get("alert_level", "Low"),
        "top_risk_drivers": data.get("top_risk_drivers", []),
        "graph_risk_score": graph_score,
        "graph_alert_level": graph_alert,
        "graph_interpretation": graph_interpretation,
        "connected_applications": graph_result.get("connected_applications", 0),
        "high_risk_connections": graph_result.get("high_risk_connections", 0),
        "declined_connections": graph_result.get("declined_connections", 0),
        "shared_risk_nodes": graph_result.get("shared_risk_nodes", []),
    }