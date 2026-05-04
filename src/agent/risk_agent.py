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
        "analysis": result["llm_analysis"]
    }