from src.agent.investigation_agent import run_investigation


def risk_decision_agent(application_reference: str) -> dict:
    result = run_investigation(application_reference)

    if "error" in result:
        return result

    data = result["data"]

    fraud = data.get("fraud_score", 0)
    risk = data.get("risk_probability", 0)

    if fraud >= 0.9:
        action = "BLOCK"
    elif fraud >= 0.75:
        action = "REVIEW"
    elif risk >= 0.6:
        action = "ESCALATE"
    else:
        action = "APPROVE"

    return {
        "application_reference": application_reference,
        "recommended_action": action,
        "recommendation": action,
        "analysis": result.get("llm_analysis", "No analysis available."),
        "llm_summary": result.get("llm_analysis", "No analysis available."),
        "investigation_summary": result.get("llm_analysis", "No investigation summary available."),
        "final_decision": data.get("final_decision", "UNKNOWN"),
        "fraud_score": data.get("fraud_score", 0),
        "risk_probability": data.get("risk_probability", 0),
        "probability_default": data.get("probability_default", 0),
        "credit_score": data.get("credit_score", 0),
        "transaction_amount": max(
            data.get("requested_amount", 0) or 0,
            data.get("approved_amount", 0) or 0,
            data.get("approved_limit", 0) or 0,
        ),
        "alert_level": data.get("alert_level", "Low"),
        "top_risk_drivers": data.get("top_risk_drivers", []),
    }