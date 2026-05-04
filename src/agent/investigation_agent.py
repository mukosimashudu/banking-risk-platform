from src.agent.tools import fetch_application
from src.agent.prompts import investigation_prompt
from src.llm.llm_assistant import generate_explanation


def run_investigation(application_reference: str) -> dict:
    data = fetch_application(application_reference)

    if not data:
        return {"error": "Application not found"}

    prompt = investigation_prompt(data)

    try:
        llm_output = generate_explanation({"prompt": prompt})
    except Exception:
        llm_output = "LLM unavailable"

    return {
        "data": data,
        "llm_analysis": llm_output
    }