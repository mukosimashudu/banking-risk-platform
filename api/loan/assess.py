from fastapi import APIRouter, HTTPException
from datetime import datetime

from src.scoring.loan_engine import affordability_engine, build_amortisation_schedule
from src.scoring.ifrs9_engine import ifrs9_engine
from src.scoring.decision_engine import make_decision
from src.scoring.explainability import explain_model
from src.llm.llm_assistant import generate_explanation

router = APIRouter()


@router.post("/api/loan/assess")
def assess_loan(payload):

    try:
        # =========================
        # 1. LOAN ENGINE
        # =========================
        result = affordability_engine(
            net_monthly_income=payload["net_monthly_income"],
            monthly_expenses=payload["monthly_expenses"],
            existing_debt_payments=payload["existing_debt_payments"],
            requested_amount=payload["requested_amount"],
            annual_interest_rate=payload["annual_interest_rate"],
            term_months=payload["term_months"],
            property_value=payload.get("property_value"),
            deposit=payload.get("deposit"),
        )

        # =========================
        # 2. IFRS9
        # =========================
        ifrs9 = ifrs9_engine(
            approved_amount=result.approved_amount,
            credit_score=payload["credit_score"],
            product_type=payload["product_type"],
            secured=payload.get("secured", False),
            days_past_due=payload.get("days_past_due", 0),
            significant_increase_in_credit_risk=payload.get("sicr_flag", False),
            default_flag=payload.get("default_flag", False),
            ltv=result.ltv,
        )

        # =========================
        # 3. DECISION
        # =========================
        decision, reason = make_decision(
            affordability=result,
            ifrs9=ifrs9,
            credit_score=payload["credit_score"],
            fraud_score=payload["fraud_score"],
        )

        # =========================
        # 4. SHAP
        # =========================
        shap_result = explain_model(payload)

        # =========================
        # 5. OPENAI
        # =========================
        llm_text = generate_explanation({
            "income": payload["net_monthly_income"],
            "expenses": payload["monthly_expenses"],
            "debt": payload["existing_debt_payments"],
            "credit_score": payload["credit_score"],
            "requested": result.requested_amount,
            "approved": result.approved_amount,
            "monthly_payment": result.monthly_payment,
            "pd": ifrs9.pd_12m,
            "lgd": ifrs9.lgd,
            "ecl": ifrs9.ecl_lifetime,
            "stage": ifrs9.stage,
            "decision": decision,
            "reason": reason,
        })

        # =========================
        # 6. SCHEDULE
        # =========================
        schedule = build_amortisation_schedule(
            result.approved_amount,
            payload["annual_interest_rate"],
            payload["term_months"],
        )

        # =========================
        # RESPONSE
        # =========================
        return {
            "decision": decision,
            "reason": reason,
            "approved_amount": result.approved_amount,
            "monthly_payment": result.monthly_payment,
            "ecl": ifrs9.ecl_lifetime,
            "stage": ifrs9.stage,
            "shap": shap_result,
            "llm_explanation": llm_text,
            "schedule": schedule
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))