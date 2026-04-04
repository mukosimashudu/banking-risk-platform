from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from src.config.db import engine
from openai import OpenAI
import os

app = FastAPI(title="Full Fintech Banking Platform API")

# =========================
# OPENAI SETUP
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_explanation(context: dict, shap_features: list) -> str:
    try:
        shap_text = "\n".join(
            [f"- {item['feature']}: impact {item['shap_value']}" for item in shap_features]
        )

        prompt = f"""
You are a senior banking credit risk analyst preparing a credit committee note.

Customer Profile:
- Credit Score: {context.get("credit_score")}
- Income: {context.get("income")}
- Debt: {context.get("debt")}
- Decision: {context.get("decision")}
- Risk Probability: {context.get("risk")}

Model Explainability (SHAP):
{shap_text}

Write a concise professional explanation covering:
1. Why the decision was made
2. Risk interpretation
3. Main drivers from the SHAP output
4. Financial behaviour insight
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a banking risk expert."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM unavailable: {str(e)}"


class LoanOfficerChatRequest(BaseModel):
    question: str
    applicant_name: str | None = None
    product_type: str | None = None
    credit_score: int | None = None
    net_monthly_income: float | None = None
    existing_debt_payments: float | None = None
    requested_amount: float | None = None
    approved_amount: float | None = None
    risk_probability: float | None = None
    final_decision: str | None = None
    fraud_alert_level: str | None = None
    top_features: list[dict] | None = None


def generate_loan_officer_chat(request: LoanOfficerChatRequest) -> str:
    try:
        top_features = request.top_features or []
        shap_text = "\n".join(
            [
                f"- {item.get('feature', 'unknown')}: impact {item.get('shap_value', 0)}"
                for item in top_features
            ]
        ) or "No SHAP features provided."

        prompt = f"""
You are an experienced loan officer assistant helping a banker explain and interpret a lending decision.

Applicant context:
- Applicant Name: {request.applicant_name}
- Product Type: {request.product_type}
- Credit Score: {request.credit_score}
- Net Monthly Income: {request.net_monthly_income}
- Existing Debt Payments: {request.existing_debt_payments}
- Requested Amount: {request.requested_amount}
- Approved Amount: {request.approved_amount}
- Risk Probability: {request.risk_probability}
- Final Decision: {request.final_decision}
- Fraud Alert Level: {request.fraud_alert_level}

Explainability inputs:
{shap_text}

User question:
{request.question}

Instructions:
- Answer in a professional but easy-to-understand banking tone.
- Be practical and specific.
- If the question asks why the application was approved or declined, refer to affordability, score, debt burden, risk, and fraud context when relevant.
- If information is missing, say so clearly.
- Keep the answer concise but useful.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a banking lending assistant for loan officers."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Loan officer assistant unavailable: {str(e)}"


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
# EXECUTIVE SUMMARY
# =========================
@app.get("/api/portfolio/summary")
def get_summary():
    try:
        with engine.connect() as conn:
            loan_count = conn.execute(
                text("SELECT COUNT(*) FROM dbo.loan_applications")
            ).scalar() or 0

            credit_count = conn.execute(
                text("SELECT COUNT(*) FROM dbo.credit_applications")
            ).scalar() or 0

            total_applications = loan_count + credit_count

            loan_approved = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM dbo.loan_applications
                    WHERE final_decision = 'APPROVED'
                    """
                )
            ).scalar() or 0

            credit_approved = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM dbo.credit_applications
                    WHERE final_decision = 'APPROVED'
                    """
                )
            ).scalar() or 0

            total_approved = loan_approved + credit_approved
            approval_rate = total_approved / total_applications if total_applications else 0

            loan_exposure = conn.execute(
                text("SELECT SUM(approved_amount) FROM dbo.loan_applications")
            ).scalar() or 0

            credit_exposure = conn.execute(
                text("SELECT SUM(approved_limit) FROM dbo.credit_applications")
            ).scalar() or 0

            avg_pd = conn.execute(
                text("SELECT AVG(risk_probability) FROM dbo.credit_applications")
            ).scalar() or 0

            avg_shap = conn.execute(
                text("SELECT AVG(shap_risk_probability) FROM dbo.loan_applications")
            ).scalar() or 0

            fraud = conn.execute(
                text("SELECT COUNT(*) FROM dbo.train_transaction WHERE isFraud = 1")
            ).scalar() or 0

            total_tx = conn.execute(
                text("SELECT COUNT(*) FROM dbo.train_transaction")
            ).scalar() or 1

            fraud_rate = fraud / total_tx if total_tx > 0 else 0

        product_distribution = [
            {"product": "Loans", "count": loan_count},
            {"product": "Credit", "count": credit_count},
        ]

        decision_distribution = [
            {"decision": "Approved", "count": total_approved},
            {"decision": "Declined", "count": total_applications - total_approved},
        ]

        fraud_distribution = [
            {"alert_level": "Fraud", "count": fraud},
            {"alert_level": "Normal", "count": total_tx - fraud},
        ]

        return {
            "total_applications": total_applications,
            "total_approved_cases": total_approved,
            "approval_rate": approval_rate,
            "total_lifetime_ecl": 0,
            "total_approved_amount": loan_exposure,
            "total_credit_limit": credit_exposure,
            "average_pd_12m": avg_pd,
            "average_fraud_score": fraud_rate,
            "critical_alerts": fraud,
            "high_alerts": int(fraud * 0.5),
            "average_shap_risk_probability": avg_shap,
            "product_distribution": product_distribution,
            "decision_distribution": decision_distribution,
            "fraud_distribution": fraud_distribution,
        }

    except Exception as e:
        print("🔥 SUMMARY ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# RECENT TRANSACTIONS (FRAUD DATA)
# =========================
@app.get("/api/portfolio/recent")
def get_recent():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT TOP 50
                        TransactionID AS application_reference,
                        TRY_CAST(TransactionAmt AS FLOAT) AS requested_amount,
                        isFraud AS fraud_flag,
                        ProductCD AS product_type
                    FROM dbo.train_transaction
                    ORDER BY TransactionID DESC
                    """
                )
            )
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
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
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
                    """
                )
            )
            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 FRAUD ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


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

        shap_features = [
            {"feature": "credit_score", "shap_value": round((650 - score) / 100, 3)},
            {"feature": "debt_to_income", "shap_value": round(dti, 3)},
            {"feature": "income", "shap_value": round(-income / 100000, 3)},
        ]

        llm_text = generate_explanation(
            {
                "credit_score": score,
                "income": income,
                "debt": debt,
                "decision": decision,
                "risk": risk_prob,
            },
            shap_features,
        )

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
                "top_features": shap_features,
            },
            "amortisation_schedule": [],
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

        shap_features = [
            {"feature": "credit_score", "shap_value": round((650 - score) / 100, 3)},
            {"feature": "debt_to_income", "shap_value": round(dti, 3)},
            {"feature": "income", "shap_value": round(-income / 100000, 3)},
        ]

        llm_text = generate_explanation(
            {
                "credit_score": score,
                "income": income,
                "debt": debt,
                "decision": decision,
                "risk": risk,
            },
            shap_features,
        )

        return {
            "final_decision": decision,
            "approved_limit": limit,
            "risk_probability": risk,
            "decision_reason": "Credit scoring logic",
            "llm_explanation": llm_text,
            "shap_explanation": {
                "available": True,
                "risk_probability": risk,
                "top_features": shap_features,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# RECENT LOANS
# =========================
@app.get("/api/loan/recent")
def get_loans():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT TOP 50 *
                    FROM dbo.loan_applications
                    ORDER BY created_at DESC
                    """
                )
            )
            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 LOAN RECENT ERROR:", str(e))
        return []


# =========================
# RECENT CREDIT
# =========================
@app.get("/api/credit/recent")
def get_credit():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT TOP 50 *
                    FROM dbo.credit_applications
                    ORDER BY created_at DESC
                    """
                )
            )
            return [dict(row._mapping) for row in result]

    except Exception as e:
        print("🔥 CREDIT RECENT ERROR:", str(e))
        return []


# =========================
# LOAN OFFICER CHAT
# =========================
@app.post("/api/loan-officer/chat")
def loan_officer_chat(payload: dict):
    try:
        question = payload.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is required.")

        applicant_name = payload.get("applicant_name")
        product_type = payload.get("product_type")
        credit_score = payload.get("credit_score")
        net_monthly_income = payload.get("net_monthly_income")
        existing_debt_payments = payload.get("existing_debt_payments")
        requested_amount = payload.get("requested_amount")
        approved_amount = payload.get("approved_amount")
        risk_probability = payload.get("risk_probability")
        final_decision = payload.get("final_decision")
        fraud_alert_level = payload.get("fraud_alert_level")
        top_features = payload.get("top_features", [])

        shap_text = "\n".join(
            [
                f"- {item.get('feature', 'unknown')}: impact {item.get('shap_value', 0)}"
                for item in top_features
            ]
        ) or "No explainability features were provided."

        prompt = f"""
You are a professional loan officer assistant helping explain a lending decision.

Application context:
- Applicant Name: {applicant_name}
- Product Type: {product_type}
- Credit Score: {credit_score}
- Net Monthly Income: {net_monthly_income}
- Existing Debt Payments: {existing_debt_payments}
- Requested Amount: {requested_amount}
- Approved Amount: {approved_amount}
- Risk Probability: {risk_probability}
- Final Decision: {final_decision}
- Fraud Alert Level: {fraud_alert_level}

Explainability inputs:
{shap_text}

User question:
{question}

Instructions:
- Answer in plain but professional banking language.
- Be specific to the provided application.
- Refer to affordability, credit score, risk, fraud alert, and SHAP features when relevant.
- If something is missing, say so clearly.
- Keep the answer concise and useful for a banker speaking to a manager or customer.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a banking lending assistant for loan officers."},
                {"role": "user", "content": prompt},
            ],
        )

        return {"answer": response.choices[0].message.content.strip()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))