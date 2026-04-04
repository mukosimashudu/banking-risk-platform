from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from src.config.db import engine
from openai import OpenAI
import os
import uuid

# =========================
# OPENAI SETUP
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_explanation(context: dict) -> str:
    try:
        prompt = f"""
You are a senior banking credit risk analyst.

Explain the decision professionally.

Customer:
- Credit Score: {context.get("credit_score")}
- Income: {context.get("income")}
- Debt: {context.get("debt")}
- Decision: {context.get("decision")}
- Risk: {context.get("risk")}

Explain clearly why the decision was made.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a banking risk expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM unavailable: {str(e)}"


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
# LOAN APPLICATION
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
        # SHAP
        # =========================
        shap_features = [
            {"feature": "credit_score", "shap_value": round((650 - score) / 100, 3)},
            {"feature": "debt_to_income", "shap_value": round(dti, 3)},
            {"feature": "income", "shap_value": round(-income / 100000, 3)},
        ]

        # =========================
        # REAL LLM
        # =========================
        llm_text = generate_explanation({
            "credit_score": score,
            "income": income,
            "debt": debt,
            "decision": decision,
            "risk": risk_prob
        })

        return {
            "final_decision": decision,
            "approved_amount": approved_amount,
            "monthly_payment": monthly_payment,
            "ecl_lifetime": 0,
            "decision_reason": "Risk-based decision",

            "llm_explanation": llm_text,

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
# CREDIT APPLICATION
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
        # REAL LLM
        # =========================
        llm_text = generate_explanation({
            "credit_score": score,
            "income": income,
            "debt": debt,
            "decision": decision,
            "risk": risk
        })

        return {
            "final_decision": decision,
            "approved_limit": limit,
            "risk_probability": risk,
            "decision_reason": "Credit scoring logic",

            "llm_explanation": llm_text,

            "shap_explanation": {
                "available": True,
                "risk_probability": risk,
                "top_features": shap_features
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))